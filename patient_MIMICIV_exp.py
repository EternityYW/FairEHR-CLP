import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from scipy.stats import wasserstein_distance


df = pd.read_csv("structured_MIMICIV.csv")
df_notes = pd.read_csv("unstructured_MIMICIV.csv")

df_demographics = df[['subject_id', 'gender', 'ethnic_group', 'race', 'age', 'insurance', 'readmission_30_days_label']]
df_demographics = df_demographics.drop_duplicates()
df_longitudinal = df[['subject_id', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2',
       'Anion gap', 'Arterial Base Excess', 'Arterial CO2 Pressure',
       'Arterial O2 pressure', 'BUN', 'Calcium non-ionized',
       'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)',
       'Glucose finger stick (range 70-100)', 'HCO3 (serum)',
       'Hematocrit (serum)', 'Hemoglobin', 'Magnesium', 'PH (Arterial)',
       'Phosphorous', 'Platelet Count', 'Potassium (serum)', 'Sodium (serum)',
       'WBC']]

def generate_different_category(original, categories):
    """Select a different category than the original."""
    synthetic = np.random.choice(categories)
    while synthetic == original:
        synthetic = np.random.choice(categories)
    return synthetic

synthetic_sex = df_demographics['gender'].apply(lambda x: generate_different_category(x, [0, 1]))
synthetic_ethnic_group = df_demographics['ethnic_group'].apply(lambda x: generate_different_category(x, [0, 1, 2]))
synthetic_race = df_demographics['race'].apply(lambda x: generate_different_category(x, [0, 1, 2, 3, 4]))
synthetic_coverage = df_demographics['insurance'].apply(lambda x: generate_different_category(x, [0, 1, 2]))


# Create synthetic data for age ensuring it stays within the range of 50-100
# We will randomly select an age within the range that is different from the original age
age_groups = {
    '50-60': range(50, 61),
    '60-70': range(60, 71),
    '70-80': range(70, 81),
    '80-90': range(80, 91),
    '90-100': range(90, 101)
}

def get_age_group(age):
    for group, ages in age_groups.items():
        if age in ages:
            return group
    return None

def get_synthetic_age(real_age_group):
    other_groups = [group for group in age_groups if group != real_age_group]
    selected_group = random.choice(other_groups)
    return random.choice(list(age_groups[selected_group]))

synthetic_age = df_demographics['age'].apply(lambda x: get_synthetic_age(get_age_group(x)))

#age_range = list(range(50, 101))
#synthetic_age = df_demographics['age'].apply(lambda x: generate_different_category(x, age_range))

# Compile the synthetic demographic DataFrame
df_synthetic_demographics = pd.DataFrame({
    'gender': synthetic_sex,
    'ethnic_group': synthetic_ethnic_group,
    'race': synthetic_race,
    'age': synthetic_age,
    'insurance': synthetic_coverage,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

longitudinal_features = df.loc[:, 'heart_rate':'WBC']
longitudinal_features_array = longitudinal_features.to_numpy()

data_min = longitudinal_features_array.min(axis=(0, 1), keepdims=True)
data_max = longitudinal_features_array.max(axis=(0, 1), keepdims=True)

# Normalize data to [-1, 1]
normalized_data = (longitudinal_features_array - data_min) / (data_max - data_min) * 2 - 1

num_patients = 15918
num_timepoints = 12
num_features = 27

data_reshaped = normalized_data.reshape((num_patients, num_timepoints, num_features))

# Convert the NumPy array to a PyTorch tensor and send to device
data_tensor = torch.tensor(data_reshaped, dtype=torch.float32).to(device)

# Flatten the tensor to fit the GAN input shape: [num_patients * num_timepoints, num_features]
data_tensor_flat = data_tensor.view(num_patients * num_timepoints, num_features)

# Define the GAN's Generator and Discriminator architectures
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, patient_data):
        return self.model(patient_data)

# Initialize the Generator and Discriminator
z_dim = 100
generator = Generator(input_size=z_dim, output_size=num_features).to(device)
generator = torch.nn.DataParallel(generator)
discriminator = Discriminator().to(device)
discriminator = torch.nn.DataParallel(discriminator)


# Set up optimizers for both G and D
# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.9)


# Binary cross entropy loss and DataLoader
criterion = nn.BCELoss()
dataloader = DataLoader(TensorDataset(data_tensor_flat), batch_size=64, shuffle=True)

# Training loop for the GAN
epochs = 10
for epoch in range(epochs):
    for i, (patients_data,) in enumerate(dataloader):
        real_data = patients_data
        real_labels = torch.full((patients_data.size(0), 1), 0.9, device=device)

        # Generate fake data and labels
        z = torch.randn(patients_data.size(0), z_dim, device=device)
        fake_data = generator(z)
        fake_labels = torch.full((patients_data.size(0), 1), 0.1, device=device)

        # Train the discriminator on real data
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_data), real_labels)
        real_loss.backward()

        # Train the discriminator on fake data
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        fake_loss.backward()
        optimizer_D.step()
        scheduler_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        generator_loss = criterion(discriminator(fake_data), real_labels)
        generator_loss.backward()
        optimizer_G.step()
        scheduler_G.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {real_loss + fake_loss} | G Loss: {generator_loss}")

# Generate synthetic data for the entire dataset
z = torch.randn(num_patients * num_timepoints, z_dim, device=device)
synthetic_data_flat = generator(z)

# Optionally reshape it to the original data format
synthetic_longitudinal_normalized = synthetic_data_flat.view(num_patients, num_timepoints, num_features).detach().cpu().numpy()

# Denormalize the synthetic data back to the original feature ranges
synthetic_longitudinal = synthetic_longitudinal_normalized * (data_max - data_min) / 2 + (data_max + data_min) / 2

synthetic_longitudinal = torch.tensor(synthetic_longitudinal).float()
df_real_demographics = df_demographics[['gender', 'ethnic_group', 'race', 'age', 'insurance']]
real_demographics = torch.tensor(df_real_demographics.values).float()
synthetic_demographics = torch.tensor(df_synthetic_demographics.values).float()
real_longitudinal = torch.tensor(longitudinal_features_array.reshape((num_patients, num_timepoints, num_features))).float()

# synthetic note obtained from Llama2 generation using notes_for_llama2.py

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.to(device)
model = torch.nn.DataParallel(model)

# Function to create embeddings for a batch of texts
def create_embeddings(texts):
    # Tokenize and prepare the texts as BERT input format
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the output of the first token ([CLS] token) for sentence embedding
    return outputs.last_hidden_state[:, 0, :]

# Process the 'notes' column in batches and collect embeddings
batch_size = 32  # You can adjust the batch size depending on your memory availability
embeddings = []
for start_idx in range(0, len(df_notes['note']), batch_size):
    batch_texts = df_notes['note'][start_idx:start_idx + batch_size].tolist()
    batch_embeddings = create_embeddings(batch_texts)
    embeddings.append(batch_embeddings)

# Concatenate all batch embeddings into a single tensor
real_notes = torch.cat(embeddings, dim=0)

embeddings = []
for start_idx in range(0, len(df_notes['synthetic_note']), batch_size):
    batch_texts = df_notes['synthetic_note'][start_idx:start_idx + batch_size].tolist()
    batch_embeddings = create_embeddings(batch_texts)
    embeddings.append(batch_embeddings)

# Concatenate all batch embeddings into a single tensor
synthetic_notes = torch.cat(embeddings, dim=0)

binary_labels = torch.tensor(df_demographics['readmission_30_days_label'].values).unsqueeze(1)

class PatientPairDataset(Dataset):
    def __init__(self, real_data, synthetic_data, labels, use_synthetic=True):
        # Initialization with real and synthetic data, and labels
        self.real_demographics, self.real_longitudinal, self.real_notes = real_data
        self.synthetic_demographics, self.synthetic_longitudinal, self.synthetic_notes = synthetic_data
        self.labels = labels
        self.use_synthetic = use_synthetic  # Control flag for using synthetic data

        # Ensure all components have the same length
        assert len(self.real_demographics) == len(self.real_longitudinal) == len(self.real_notes) == \
               len(self.synthetic_demographics) == len(self.synthetic_longitudinal) == len(self.synthetic_notes) == \
               len(self.labels), "All components of the dataset must have the same length."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return data based on the use_synthetic flag
        if self.use_synthetic:
            # For training, return both real and synthetic data
            return (self.real_demographics[idx], self.real_longitudinal[idx], self.real_notes[idx],
                    self.synthetic_demographics[idx], self.synthetic_longitudinal[idx], self.synthetic_notes[idx],
                    self.labels[idx])
        else:
            # For testing, return only real data
            return (self.real_demographics[idx], self.real_longitudinal[idx], self.real_notes[idx], self.labels[idx])
'''
def split_indices(dataset, train_ratio=0.8):
    # Function to split the dataset into train and test indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(train_ratio * dataset_size)
    random.shuffle(indices)
    return indices[:split], indices[split:]
'''

def stratified_split_indices(labels, train_ratio=0.8):
    # Assuming labels is a list or numpy array of binary labels
    indices = list(range(len(labels)))
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio, stratify=labels)
    return train_indices, test_indices

# Assuming real_data, synthetic_data, and labels are defined
real_data = (real_demographics, real_longitudinal, real_notes)
synthetic_data = (synthetic_demographics, synthetic_longitudinal, synthetic_notes)
labels = binary_labels

full_dataset = PatientPairDataset(real_data, synthetic_data, labels)

# Split indices for train and test sets
#train_indices, test_indices = split_indices(full_dataset)
train_indices, test_indices = stratified_split_indices(labels, train_ratio=0.8)


# Create train and test datasets
train_dataset = Subset(PatientPairDataset(real_data, synthetic_data, labels, use_synthetic=True), train_indices)
test_dataset = Subset(PatientPairDataset(real_data, synthetic_data, labels, use_synthetic=False), test_indices)

# DataLoaders for the train and test datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
class DynamicRelevanceBiasMitigationLayer(nn.Module):
    """
    Dynamically adjusts the influence of different data types to mitigate potential biases.
    """
    def __init__(self, input_size):
        super().__init__()
        self.adjustment_weights = nn.Parameter(torch.randn(input_size))

    def forward(self, combined_input):
        # Use broadcasting for adjustment weights without explicit expansion
        adjusted_weights = torch.sigmoid(self.adjustment_weights)
        adjusted_output = combined_input * adjusted_weights
        return adjusted_output


class FairnessAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Assuming each branch processes its input and reduces it to a 32-dimensional output
        self.demographics_branch = nn.Sequential(nn.Linear(5, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.longitudinal_branch = nn.Sequential(nn.Linear(12*27, 32), nn.BatchNorm1d(32), nn.ReLU()) 
        self.notes_branch = nn.Sequential(nn.Linear(768, 32), nn.BatchNorm1d(32), nn.ReLU())
        
        # DRBM Layer initialization
        self.drbm_layer = DynamicRelevanceBiasMitigationLayer(32)
        
        # Fusion layer to combine features from the three branches
        self.fusion_layer = nn.Sequential(nn.Linear(32 * 3, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU())

        # Output layer for embeddings and classification
        self.embedding_layer = nn.Sequential(nn.Linear(32, 16), nn.BatchNorm1d(16))
        self.classifier = nn.Linear(16, 2)  # Output logits for 2 classes (binary classification)

    def forward(self, real_demographics, real_longitudinal, real_notes, synthetic_demographics, synthetic_longitudinal, synthetic_notes):
        # Process real and synthetic data
        # Flatten longitudinal data
        real_longitudinal_flat = real_longitudinal.view(real_longitudinal.size(0), -1)
        synthetic_longitudinal_flat = synthetic_longitudinal.view(synthetic_longitudinal.size(0), -1)

        real_demo_repr = self.demographics_branch(real_demographics)
        real_long_repr = self.longitudinal_branch(real_longitudinal_flat)
        real_notes_repr = self.notes_branch(real_notes)

        synthetic_demo_repr = self.demographics_branch(synthetic_demographics)
        synthetic_long_repr = self.longitudinal_branch(synthetic_longitudinal_flat)
        synthetic_notes_repr = self.notes_branch(synthetic_notes)

        # Combine features from all branches
        real_combined = self.fusion_layer(torch.cat([real_demo_repr, real_long_repr, real_notes_repr], dim=1))
        synthetic_combined = self.fusion_layer(torch.cat([synthetic_demo_repr, synthetic_long_repr, synthetic_notes_repr], dim=1))
        
        # Apply the DRBM layer for dynamic bias mitigation
        real_adjusted = self.drbm_layer(real_combined)
        synthetic_adjusted = self.drbm_layer(synthetic_combined)

        # Process through the fusion layer and subsequent layers
        real_embedding = self.embedding_layer(real_adjusted)
        synthetic_embedding = self.embedding_layer(synthetic_adjusted)
        logits = self.classifier(real_embedding)
        classification_logits = F.softmax(logits, dim=1)

        return real_embedding, synthetic_embedding, classification_logits
    
class FairnessAwareContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.65, beta=0.35, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, real_embeddings, synthetic_embeddings):
        # Standard Contrastive Loss Component
        positive_similarity = self.cosine_similarity(real_embeddings, synthetic_embeddings)
        batch_size = real_embeddings.size(0)
        negative_similarity = sum(
            self.cosine_similarity(real_embeddings[i].unsqueeze(0), synthetic_embeddings[j].unsqueeze(0))
            for i in range(batch_size) for j in range(batch_size) if i != j
        ) / (batch_size * (batch_size - 1))
        contrastive_loss = torch.mean(torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))

        # Fairness-aware Loss Component
        fairness_loss = self.calculate_fairness_loss(real_embeddings, synthetic_embeddings)

        # Combined Loss
        combined_loss = self.beta * contrastive_loss + self.alpha * fairness_loss
        return combined_loss
    
    def calculate_fairness_loss(self, real_embeddings, synthetic_embeddings):
        # Distance Component: Euclidean distance (L2 norm)
        euclidean_distances = torch.norm(real_embeddings - synthetic_embeddings, dim=1, p=2)

        # Angle Component: Cosine similarity (cosine of angle)
        cosine_similarities = self.cosine_similarity(real_embeddings, synthetic_embeddings)

        # Convert cosine similarities to angles in radians
        angles = torch.acos(torch.clamp(cosine_similarities, -1.0, 1.0))

        # Combine distance and angle components
        # Harmonic mean of distances and angles
        combined_metric = 2 * (euclidean_distances * angles) / (euclidean_distances + angles + 1e-8)

        # Fairness loss is the mean of the combined metric
        fairness_loss = torch.mean(combined_metric)

        return fairness_loss


'''
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, real_embeddings, synthetic_embeddings):
        # Calculate positive similarity
        positive_similarity = self.cosine_similarity(real_embeddings, synthetic_embeddings)

        # Calculate negative similarities
        batch_size = real_embeddings.size(0)
        negative_similarity = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    negative_similarity += self.cosine_similarity(real_embeddings[i].unsqueeze(0), synthetic_embeddings[j].unsqueeze(0))
        negative_similarity /= (batch_size * (batch_size - 1))

        contrastive_loss = torch.mean(torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))
        return contrastive_loss
'''
    
model = FairnessAwareModel().to(device)
model = torch.nn.DataParallel(model)
contrastive_criterion = FairnessAwareContrastiveLoss().to(device)

# class_weights = torch.tensor([2.0, 1.0], dtype=torch.float) (DELIRIUM/OUD)
class_weights = torch.tensor([3.5, 0.9], dtype=torch.float)

# Move the weights to the same device as your model
class_weights = class_weights.to(device)

# Create a weighted loss function
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop

num_epochs = 10  # Define the number of epochs

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_total_contrastive_loss = 0.0
    train_total_classification_loss = 0.0
    train_correct_predictions = 0
    train_total_samples = 0

    for data in train_dataloader:
        real_demographics, real_longitudinal, real_notes, synthetic_demographics, synthetic_longitudinal, synthetic_notes, labels = [d.to(device) for d in data]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        real_embedding, synthetic_embedding, logits = model(real_demographics, real_longitudinal, real_notes, synthetic_demographics, synthetic_longitudinal, synthetic_notes)

        # Calculate the contrastive and classification loss
        contrastive_loss = contrastive_criterion(real_embedding, synthetic_embedding)
        classification_loss = classification_criterion(logits, labels.squeeze().long())
        total_loss = contrastive_loss + classification_loss

        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()

        # Accumulate losses and calculate accuracy
        train_total_contrastive_loss += contrastive_loss.item()
        train_total_classification_loss += classification_loss.item()
        _, predicted = torch.max(logits, 1)
        train_total_samples += labels.size(0)
        train_correct_predictions += (predicted == labels.squeeze().long()).sum().item()

    # Compute average training losses and accuracy
    train_avg_contrastive_loss = train_total_contrastive_loss / len(train_dataloader)
    train_avg_classification_loss = train_total_classification_loss / len(train_dataloader)
    train_accuracy = 100 * train_correct_predictions / train_total_samples

    # Testing Phase
    model.eval()
    test_total_loss = 0.0
    test_correct_predictions = 0
    test_total_samples = 0
    test_demographics, test_ground_truth, test_predictions, test_logits = [], [], [], []
    

    with torch.no_grad():
        for real_demographics, real_longitudinal, real_notes, labels in test_dataloader:
            real_demographics, real_longitudinal, real_notes, labels = real_demographics.to(device), real_longitudinal.to(device), real_notes.to(device), labels.to(device)

            # Forward pass with real data only
            real_embedding, _, logits = model(real_demographics, real_longitudinal, real_notes, real_demographics, real_longitudinal, real_notes)

            # Calculate the classification loss
            classification_loss = classification_criterion(logits, labels.squeeze().long())
            test_total_loss += classification_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            test_total_samples += labels.size(0)
            test_correct_predictions += (predicted == labels.squeeze().long()).sum().item()
            test_ground_truth.extend(labels.tolist())
            test_predictions.extend(predicted.tolist())
            test_logits.extend(logits[:, 1].detach().cpu().numpy())
            test_demographics.extend(real_demographics.detach().cpu().numpy())

    # Compute average testing loss and accuracy
    test_avg_loss = test_total_loss / len(test_dataloader)
    test_accuracy = 100 * test_correct_predictions / test_total_samples
    #test_logits_np = np.concatenate(test_logits)

    # Print training and testing statistics
    print(f"Epoch [{epoch+1}/{num_epochs}]:")
    print(f"  Training - Contrastive Loss: {train_avg_contrastive_loss:.4f}, Classification Loss: {train_avg_classification_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"  Testing - Loss: {test_avg_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    epoch_str = f"epoch{epoch+1}"
    np.save(f'{epoch_str}_readmission_test_ground_truth.npy', np.array(test_ground_truth))
    np.save(f'{epoch_str}_readmission_test_predictions.npy', np.array(test_predictions))
    np.save(f'{epoch_str}_readmission_test_logits.npy', np.array(test_logits))
    np.save(f'{epoch_str}_readmission_test_demographics.npy', np.array(test_demographics))

# Optionally, save test ground truth and predictions
#test_ground_truth_np = np.array(test_ground_truth)
#test_predictions_np = np.array(test_predictions)

# Save as .npy files
#np.save('test_ground_truth.npy', test_ground_truth_np)
#np.save('test_predictions.npy', test_predictions_np)
#np.save('test_logits.npy', test_logits)