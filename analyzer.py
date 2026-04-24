import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

df = pd.read_csv("students.csv")

print("=" * 45)
print("        STUDENT PERFORMANCE ANALYZER")
print("=" * 45)

print("\nData Loaded Successfully!")
print(df)
print("\nShape:", df.shape)

subject_avg = df[["Math", "Science", "English"]].mean()

print("\nSubject Averages:")
print(subject_avg)

df["Total"]   = df["Math"] + df["Science"] + df["English"]
df["Average"] = (df["Total"] / 3).round(2)

def get_grade(avg):
    if avg >= 90:   return "A"
    elif avg >= 75: return "B"
    elif avg >= 60: return "C"
    else:           return "D"

df["Grade"] = df["Average"].apply(get_grade)

print("\n📋 Updated Table with Total, Average & Grade:")
print(df[["Name", "Math", "Science", "English", "Total", "Average", "Grade"]])

averages = df["Average"].values

print("\n NumPy Statistics:")
print(f"  Highest Average  : {np.max(averages)}")
print(f"  Lowest Average   : {np.min(averages)}")
print(f"  Class Mean       : {np.mean(averages):.2f}")
print(f"  Std Deviation    : {np.std(averages):.2f}")


top_student    = df.loc[df["Average"].idxmax(), "Name"]
bottom_student = df.loc[df["Average"].idxmin(), "Name"]

print(f"\n Top Student    : {top_student}")
print(f"Needs Help     : {bottom_student}")


colors = [
    "green"  if avg >= 75 else
    "orange" if avg >= 60 else
    "red"
    for avg in df["Average"]
]

plt.figure(figsize=(10, 5))
bars = plt.bar(df["Name"], df["Average"], color=colors, edgecolor="black")


for bar, avg in zip(bars, df["Average"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(avg),
        ha="center", va="bottom", fontsize=9
    )

plt.axhline(y=60, color="red",   linestyle="--", linewidth=1.2, label="Pass mark (60)")
plt.axhline(y=75, color="green", linestyle="--", linewidth=1.2, label="B grade (75)")

plt.title("Average Score per Student", fontsize=14)
plt.xlabel("Student Name")
plt.ylabel("Average Score")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("output/student_performance.png")
plt.show()
print("Bar chart saved -> output/student_performance.png")

# CHART 2: Pie Chart — Subject Average Distribution

plt.figure(figsize=(6, 6))
plt.pie(
    subject_avg,
    labels=subject_avg.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#FF6B6B", "#4ECDC4", "#45B7D1"],
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
plt.title("Subject-wise Average Scores", fontsize=14)
plt.tight_layout()
plt.savefig("output/subject_averages.png")
plt.show()
print("Pie chart saved  → output/subject_averages.png")

# CHART 3: Line Chart — Attendance vs Average Score

df_sorted = df.sort_values("Attendance")

plt.figure(figsize=(8, 5))
plt.plot(
    df_sorted["Attendance"], df_sorted["Average"],
    marker="o", color="green", linewidth=2, markersize=8
)

# Label each point with student name
for _, row in df_sorted.iterrows():
    plt.annotate(
        row["Name"],
        (row["Attendance"], row["Average"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

plt.title("Attendance vs Average Score", fontsize=14)
plt.xlabel("Attendance (%)")
plt.ylabel("Average Score")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("output/line_chart.png")
plt.show()
print(" Line chart saved → output/line_chart.png")


# CHART 4: Horizontal Bar Chart — Grade Distribution

grade_counts = df["Grade"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.barh(grade_counts.index, grade_counts.values,
         color=["green", "steelblue", "orange", "red"],
         edgecolor="black")

for i, val in enumerate(grade_counts.values):
    plt.text(val + 0.05, i, str(val), va="center", fontsize=10)

plt.title("Grade Distribution", fontsize=14)
plt.xlabel("Number of Students")
plt.ylabel("Grade")
plt.tight_layout()
plt.savefig("output/grade_distribution.png")
plt.show()
print(" Grade chart saved → output/grade_distribution.png")



print("\n" + "=" * 45)
print("               FINAL SUMMARY")
print("=" * 45)
print(f"  Total Students   : {len(df)}")
print(f"  Class Average    : {np.mean(averages):.2f}")
print(f"  Top Student      : {top_student}")
print(f"  Needs Help       : {bottom_student}")
print(f"  Grade A students : {len(df[df['Grade'] == 'A'])}")
print(f"  Grade D students : {len(df[df['Grade'] == 'D'])}")
print("=" * 45)
