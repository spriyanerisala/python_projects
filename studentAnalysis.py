import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
students = 30

math_marks= np.random.randint(40, 100, students)  
science_marks = np.random.randint(40, 100, students)
english_marks = np.random.randint(40, 100, students)

names = [f"student_{i+1}" for i in range(students)]

df = pd.DataFrame({
    "Name"   : names,
    "Math"   : math_marks,
    "Science": science_marks,
    "English": english_marks
})

df["Total"]   = df["Math"] + df["Science"] + df["English"]
df["Average"] = (df["Total"] / 3).round(2)

def get_grade(avg):
    if avg >= 85:   return "A"
    elif avg >= 70: return "B"
    elif avg >= 55: return "C"
    else:           return "D"

df["Grade"] = df["Average"].apply(get_grade)

print("=" * 50)
print("Student Data Analysis Report")
print("=" * 50)

print(df.head())

print("\nSubject-Wise Statistics")
print(df[["Math", "Science", "English"]].describe().round(2))  

print("\nTop 5 Students")
top5 = df.nlargest(5, "Average")[["Name", "Math", "Science", "English", "Average", "Grade"]]
print(top5.to_string(index=False))                           

print("\nGrade Distribution")
print(df["Grade"].value_counts().to_string())

print("\nSubject Averages")                                    
print(f"Math    : {df['Math'].mean():.2f}")                     
print(f"Science : {df['Science'].mean():.2f}")
print(f"English : {df['English'].mean():.2f}")

pass_count = (df["Average"] >= 50).sum()
fail_count = students - pass_count
print(f"\nPass : {pass_count}")
print(f"Fail : {fail_count}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Data Analysis Dashboard", fontsize=16, fontweight="bold")

ax1 = axes[0, 0]
subjects = ["Math", "Science", "English"]
averages = [df["Math"].mean(), df["Science"].mean(), df["English"].mean()]
colors   = ["#4C72B0", "#55A868", "#C44E52"]
bars = ax1.bar(subjects, averages, color=colors, edgecolor="black", width=0.5)
ax1.set_title("Subject-wise Average Marks", fontweight="bold")
ax1.set_ylabel("Average Marks")
ax1.set_ylim(0, 100)
for bar, val in zip(bars, averages):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

ax2 = axes[0, 1]
grade_counts = df["Grade"].value_counts()
pie_colors   = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
ax2.pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%",
        colors=pie_colors[:len(grade_counts)], startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2})
ax2.set_title("Grade Distribution", fontweight="bold")

ax3 = axes[1, 0]
scatter = ax3.scatter(df["Math"], df["Science"],
                      c=df["Average"], cmap="RdYlGn",
                      s=80, edgecolors="black", linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label="Average Marks")
ax3.set_title("Math vs Science Marks", fontweight="bold")
ax3.set_xlabel("Math Marks")
ax3.set_ylabel("Science Marks")
m, b = np.polyfit(df["Math"], df["Science"], 1)
x_line = np.linspace(df["Math"].min(), df["Math"].max(), 100)
ax3.plot(x_line, m * x_line + b, color="red", linestyle="--", linewidth=1.5, label="Trend")
ax3.legend()

ax4 = axes[1, 1]
top10 = df.nlargest(10, "Average")[["Name", "Average"]].reset_index(drop=True)
bar_colors = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2
              else "#6baed6" for i in range(len(top10))]
ax4.barh(top10["Name"], top10["Average"], color=bar_colors, edgecolor="black")
ax4.set_title("Top 10 Students by Average", fontweight="bold")
ax4.set_xlabel("Average Marks")
ax4.set_xlim(0, 105)
ax4.invert_yaxis()
for i, val in enumerate(top10["Average"]):
    ax4.text(val + 0.5, i, f"{val}", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("student_analysis_charts.png", dpi=150, bbox_inches="tight")
plt.show()
