import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== STEP 1: LOAD & EXPLORE ====================
print("="*70)
print("STUDENT PERFORMANCE ANALYSIS - STEP 1: DATA EXPLORATION")
print("="*70)

df = pd.read_csv('student_dataset_10000_rows.csv')

# 1. Check dataset shape
print("\n1️⃣ DATASET SHAPE:")
print(f"   Total Records: {df.shape[0]}")
print(f"   Total Columns: {df.shape[1]}")

# 2. View first few rows
print("\n2️⃣ FIRST 5 ROWS:")
print(df.head())

# 3. Check column info
print("\n3️⃣ COLUMN INFORMATION:")
print(df.info())

# 4. Basic statistics
print("\n4️⃣ STATISTICAL SUMMARY:")
print(df.describe().round(2))

# 5. Missing values
print("\n5️⃣ MISSING VALUES:")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("   ✅ No missing values - Data is CLEAN!")

# 6. Duplicates check
print(f"\n6️⃣ DUPLICATE ROWS: {df.duplicated().sum()}")
if df.duplicated().sum() == 0:
    print("   ✅ No duplicates - Data is CLEAN!")

# ==================== STEP 2: DATA CLEANING ====================
print("\n" + "="*70)
print("STEP 2: DATA CLEANING")
print("="*70)

# Remove any missing values (if they exist)
df = df.dropna()
print(f"After removing NaN: {df.shape}")

# Remove duplicates (if they exist)
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")
print("✅ Data cleaned successfully!")

# ==================== STEP 3: CORRELATION ANALYSIS ====================
print("\n" + "="*70)
print("STEP 3: CORRELATION ANALYSIS")
print("="*70)

# Calculate correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

print("\n📊 CORRELATION WITH EXAM SCORE (Performance):")
print("-" * 50)
exam_corr = correlation['exam_score'].sort_values(ascending=False)
for var, val in exam_corr.items():
    bar_length = int(abs(val) * 30)
    bar = '█' * bar_length
    print(f"{var:30s} {val:7.3f}  {bar}")

# ==================== STEP 4: KEY INSIGHTS ====================
print("\n" + "="*70)
print("STEP 4: KEY INSIGHTS")
print("="*70)

print("\n📈 EXAM SCORE STATISTICS:")
print(f"   Mean:       {df['exam_score'].mean():.2f}")
print(f"   Median:     {df['exam_score'].median():.2f}")
print(f"   Std Dev:    {df['exam_score'].std():.2f}")
print(f"   Min:        {df['exam_score'].min():.2f}")
print(f"   Max:        {df['exam_score'].max():.2f}")

print("\n👥 PLACEMENT STATUS:")
placement = df['placement_status'].value_counts()
for status, count in placement.items():
    percentage = (count / len(df)) * 100
    print(f"   {status}: {count} students ({percentage:.1f}%)")

print("\n📚 STUDY HABITS:")
print(f"   Avg Study Hours:        {df['study_hours'].mean():.2f}")
print(f"   Avg Attendance:         {df['attendance'].mean():.2f}%")
print(f"   Avg Sleep Hours:        {df['sleep_hours'].mean():.2f}")
print(f"   Avg Internet Usage:     {df['internet_usage'].mean():.2f} hours")
print(f"   Avg Assignments Done:   {df['assignments_completed'].mean():.2f}")
print(f"   Avg Previous Score:     {df['previous_score'].mean():.2f}")

# ==================== STEP 5: VISUALIZATION SETUP ====================
print("\n" + "="*70)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("\n🎨 Creating Dashboard...")