import csv
import random
import math

# Set seed for reproducibility (optional)
random.seed(42)

# Generate data with non-linear separability
# Pattern: Label 1 = inner circle, Label 0 = outer ring
data = []

# First 58 rows with label 1 (inner circle)
for i in range(58):
    # Generate points closer to center with some noise
    radius = random.uniform(0, 0.3)  # Inner radius
    angle = random.uniform(0, 2 * math.pi)
    
    feature1 = radius * math.cos(angle) + random.gauss(0, 0.05)
    feature2 = radius * math.sin(angle) + random.gauss(0, 0.05)
    
    # Ensure values are within [-0.9, 1)
    feature1 = max(-0.9, min(0.999, feature1))
    feature2 = max(-0.9, min(0.999, feature2))
    
    data.append([feature1, feature2, 1])

# Next 60 rows (59-118) with label 0 (outer ring)
for i in range(60):
    # Generate points in outer ring with some noise
    radius = random.uniform(0.4, 0.8)  # Outer radius
    angle = random.uniform(0, 2 * math.pi)
    
    feature1 = radius * math.cos(angle) + random.gauss(0, 0.05)
    feature2 = radius * math.sin(angle) + random.gauss(0, 0.05)
    
    # Ensure values are within [-0.9, 1)
    feature1 = max(-0.9, min(0.999, feature1))
    feature2 = max(-0.9, min(0.999, feature2))
    
    data.append([feature1, feature2, 0])

# Add 4 noise points in the most confusing regions
# 2 points in inner circle region (radius ~0.3) but labeled 0 (should be 1)
for i in range(2):
    radius = random.uniform(0.25, 0.35)  # At the edge of inner circle
    angle = random.uniform(0, 2 * math.pi)
    
    feature1 = radius * math.cos(angle) + random.gauss(0, 0.03)
    feature2 = radius * math.sin(angle) + random.gauss(0, 0.03)
    
    feature1 = max(-0.9, min(0.999, feature1))
    feature2 = max(-0.9, min(0.999, feature2))
    
    data.append([feature1, feature2, 0])  # Mislabeled as 0

# 2 points in outer ring region (radius ~0.4-0.5) but labeled 1 (should be 0)
for i in range(2):
    radius = random.uniform(0.4, 0.5)  # At the inner edge of outer ring
    angle = random.uniform(0, 2 * math.pi)
    
    feature1 = radius * math.cos(angle) + random.gauss(0, 0.03)
    feature2 = radius * math.sin(angle) + random.gauss(0, 0.03)
    
    feature1 = max(-0.9, min(0.999, feature1))
    feature2 = max(-0.9, min(0.999, feature2))
    
    data.append([feature1, feature2, 1])  # Mislabeled as 1

# Write to CSV
with open('logistic_regression_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['feature1', 'feature2', 'label'])
    writer.writerows(data)

print("CSV file 'logistic_regression_data.csv' generated successfully!")
print(f"Total rows: {len(data)}")
print(f"Label 1 count: {sum(1 for row in data if row[2] == 1)}")
print(f"Label 0 count: {sum(1 for row in data if row[2] == 0)}")
print(f"Noise points added: 4 (2 inner circle mislabeled as 0, 2 outer ring mislabeled as 1)")