import os
import sys
BASE_DIR = 'e:/各种PY程序/18-医疗AI模型系统'
sys.path.insert(0, BASE_DIR)

from data.loader import HealthcareTokenLoader
from data.preprocessor import HealthcareTokenPreprocessor
from models.classifier import HealthcareTokenClassifier

loader = HealthcareTokenLoader()
df = loader.load_all(sample=100)
categories = list(loader.categories.keys())
data_types = list(loader.data_types.keys())

preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
X, y = preprocessor.fit_transform(df)

model_path = os.path.join(BASE_DIR, 'outputs', 'healthcare_token_classifier.pt')
print(f'Model file: {model_path}')
print(f'File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB')

model = HealthcareTokenClassifier.load(model_path)
print(f'Model params: input_dim={model.input_dim}, num_categories={model.num_categories}')

preds = model.predict(X)

print('\n=== Prediction Results ===')
level_ratio = preds['level_pred'].mean() * 100
print(f'Token level (A): {level_ratio:.1f}%')
quality_pred_mean = preds['quality_pred'].mean()
quality_true_mean = y['quality'].mean()
print(f'Quality score - predicted: {quality_pred_mean:.2f}, actual: {quality_true_mean:.2f}')
quality_mae = abs(preds['quality_pred'] - y['quality']).mean()
print(f'Quality MAE: {quality_mae:.2f}')
category_acc = (preds['category_pred'] == y['category']).mean() * 100
print(f'Category accuracy: {category_acc:.1f}%')

print('\n=== Category Distribution ===')
category_names = {
    'radiology': '放射科', 'pathology': '病理科', 'neurology': '神经内科',
    'cardiology': '心血管科', 'laboratory': '检验科', 'orthopedics': '骨科',
    'pediatrics': '儿科', 'emergency': '急诊科'
}
for i, cat in enumerate(categories):
    count = (preds['category_pred'] == i).sum()
    print(f'  {category_names.get(cat, cat)}: {count} records')
