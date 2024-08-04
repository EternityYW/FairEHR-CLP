import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

# Load data
df_structured = pd.read_csv('structured_MIMICIII.csv')
df_notes = pd.read_csv('unstructured_MIMICIII.csv') 

df_demographics = df_structured[['subject_id', 'gender', 'ethnic_group', 'race', 'age', 'insurance']]
df_demographics = df_demographics.drop_duplicates()
df_longitudinal = df_structured[['heartrate', 'sysbp', 'diasbp', 'meanbp', 'resprate', 'tempc', 'spo2',
       'Anion gap', 'Arterial Base Excess', 'Arterial CO2 Pressure',
       'Arterial O2 pressure', 'BUN', 'Calcium non-ionized',
       'Chloride (serum)', 'Creatinine', 'Glucose (serum)',
       'Glucose finger stick', 'HCO3 (serum)', 'Hematocrit (serum)',
       'Hemoglobin', 'Magnesium', 'PH (Arterial)', 'Phosphorous',
       'Platelet Count', 'Potassium (serum)', 'Sodium (serum)', 'WBC']]
df_labels = df_structured[['subject_id', 'readmission_30_days_label']]
df_labels = df_labels.drop_duplicates()
labels = df_labels['readmission_30_days_label'].values
notes = df_notes['note'].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.to(device)
model = torch.nn.DataParallel(model)

# Function to create embeddings for a batch of texts
def create_embeddings(texts):
    # Tokenize and prepare the texts as BERT input format
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
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

demographic_data = df_demographics.values
longitudinal_data = df_longitudinal.values.reshape((4302, 12, 27))


# Number of patients is the first dimension of the longitudinal data
num_patients = longitudinal_data.shape[0]
patient_indices = range(num_patients)

# Split indices into training and test sets
train_indices, test_indices = train_test_split(patient_indices, test_size=0.2, random_state=42, stratify=labels)

# Splitting the longitudinal data
X_long_train = longitudinal_data[train_indices]
X_long_test = longitudinal_data[test_indices]

# Assuming real_notes is a 1D array with one entry per patient
X_notes_train = real_notes[train_indices]
X_notes_test = real_notes[test_indices]

# Assuming labels is a 1D array with one entry per patient
y_train = labels[train_indices]
y_test = labels[test_indices]

# Assuming demographic_data is a DataFrame with rows corresponding to patients
demog_train = demographic_data[train_indices]
demog_test = demographic_data[test_indices]

class EHRDataset(Dataset):
    def __init__(self, longitudinal_data, notes_data, labels, demographics_data):
        self.longitudinal_data = torch.tensor(longitudinal_data, dtype=torch.float)
        self.notes_data = torch.tensor(notes_data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.demographics_data = demographics_data  # Keep as a numpy array or list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.longitudinal_data[idx], self.notes_data[idx], self.labels[idx], self.demographics_data[idx]

    
batch_size = 32  # Define your batch size

train_dataset = EHRDataset(X_long_train, X_notes_train, y_train, demog_train)
test_dataset = EHRDataset(X_long_test, X_notes_test, y_test, demog_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

'''
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
'''

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Longitudinal data branch
        self.longitudinal_branch = nn.Sequential(
            nn.Linear(12 * 27, 32),  # Adjust based on your flattened longitudinal data size
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Notes data branch
        self.notes_branch = nn.Sequential(
            nn.Linear(768, 32),  # Assuming notes data is a 768-dimensional vector
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # DRBM Layer
        #self.drbm_layer = DynamicRelevanceBiasMitigationLayer(64)  # Input size after concatenating two branches

        # Fusion layer to combine features from the longitudinal and notes branches
        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # Output layer for classification
        self.classifier = nn.Linear(16, 2)  # Binary classification

    def forward(self, longitudinal, notes):
        # Process longitudinal and notes data
        long_repr = self.longitudinal_branch(longitudinal.view(longitudinal.size(0), -1))
        notes_repr = self.notes_branch(notes)

        # Combine features from both branches
        combined = torch.cat([long_repr, notes_repr], dim=1)

        # Apply the DRBM layer for dynamic bias mitigation
        #adjusted = self.drbm_layer(combined)

        # Process through the fusion layer
        embedding = self.fusion_layer(combined)

        # Classification logits
        logits = self.classifier(embedding)
        classification_logits = F.softmax(logits, dim=1)

        return classification_logits
    
model = BaseModel().to(device)
model = torch.nn.DataParallel(model)
    
#class_weights = torch.tensor([0.8, 1.0], dtype=torch.float) (DELIRIUM)

class_weights = torch.tensor([2.5, 1.0], dtype=torch.float)
#class_weights = torch.tensor([6.0, 1.0], dtype=torch.float) (Readmission)
 
# Move the weights to the same device as your model
class_weights = class_weights.to(device)

# Create a weighted loss function
classification_criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 10

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_total_loss = 0.0
    train_correct_predictions = 0
    train_total_samples = 0

    for longitudinal, notes, labels, _ in train_loader:  # Demographics data is ignored here
        longitudinal, notes, labels = longitudinal.to(device), notes.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(longitudinal, notes)

        # Calculate the classification loss
        classification_loss = classification_criterion(logits, labels.squeeze().long())
        
        # Backward pass and optimize
        classification_loss.backward()
        optimizer.step()

        # Accumulate loss and calculate accuracy
        train_total_loss += classification_loss.item()
        _, predicted = torch.max(logits, 1)
        train_total_samples += labels.size(0)
        train_correct_predictions += (predicted == labels.squeeze().long()).sum().item()

    # Compute average training loss and accuracy
    train_avg_loss = train_total_loss / len(train_loader)
    train_accuracy = 100 * train_correct_predictions / train_total_samples

    # Testing Phase
    model.eval()
    test_total_loss = 0.0
    test_correct_predictions = 0
    test_total_samples = 0
    test_demographics, test_ground_truth, test_predictions, test_logits = [], [], [], []

    with torch.no_grad():
        for longitudinal, notes, labels, demographics in test_loader:
            longitudinal, notes, labels = longitudinal.to(device), notes.to(device), labels.to(device)

            # Forward pass
            logits = model(longitudinal, notes)

            # Calculate the classification loss
            classification_loss = classification_criterion(logits, labels.squeeze().long())
            test_total_loss += classification_loss.item()

            # Calculate accuracy and save results
            _, predicted = torch.max(logits, 1)
            test_total_samples += labels.size(0)
            test_correct_predictions += (predicted == labels.squeeze().long()).sum().item()
            test_ground_truth.extend(labels.tolist())
            test_predictions.extend(predicted.tolist())
            test_logits.extend(logits[:, 1].detach().cpu().numpy())
            test_demographics.extend(demographics.tolist())  # Save demographics for analysis

    # Compute average testing loss and accuracy
    test_avg_loss = test_total_loss / len(test_loader)
    test_accuracy = 100 * test_correct_predictions / test_total_samples

    # Print training and testing statistics
    print(f"Epoch [{epoch+1}/{num_epochs}]:")
    print(f"  Training - Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"  Testing - Loss: {test_avg_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Save test results including demographic data
    epoch_str = f"epoch{epoch+1}"
    np.save(f'{epoch_str}_test_demographics.npy', np.array(test_demographics))
    np.save(f'{epoch_str}_test_ground_truth.npy', np.array(test_ground_truth))
    np.save(f'{epoch_str}_test_predictions.npy', np.array(test_predictions))
    np.save(f'{epoch_str}_test_logits.npy', np.array(test_logits))