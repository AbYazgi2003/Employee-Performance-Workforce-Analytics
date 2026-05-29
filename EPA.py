import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

try:
    df = pd.read_csv("employees.csv")
    df = df.set_index("employee_id")

    print("\nFirst 5 Employees\n")
    print(df.head())

    print("\nDataset Information\n")
    print(df.info())

    print("\nDataset Description\n")
    print(df.describe())

    print("\nDataset Shape\n")
    print(df.shape)

    print("\nColumns\n")
    print(df.columns)

    print("\nMissing Values\n")
    print(df.isnull().sum())

    print("\nDuplicated Rows\n")
    print(df.duplicated().sum())

    df["salary"] = df["salary"].fillna(df["salary"].median())
    df["performance_score"] = df["performance_score"].fillna(
        df["performance_score"].mean()
    )

    df.loc[df["age"] > 60, "age"] = int(df["age"].median())

    df.loc[
        (df["attendance_rate"] > 100)
        | (df["attendance_rate"] < 0),
        "attendance_rate"
    ] = df["attendance_rate"].median()

    df.loc[
        (df["performance_score"] > 100)
        | (df["performance_score"] < 0),
        "performance_score"
    ] = df["performance_score"].median()

    salary_per_department = (
        df.groupby("department")["salary"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage Salary Per Department\n")
    print(salary_per_department)

    best_department = (
        df.groupby("department")["performance_score"]
        .mean()
        .idxmax()
    )

    print(f"\nBest Performing Department: {best_department}")

    resignation_rate = (
        df.groupby("department")["resigned"]
        .apply(lambda x: (x == "Yes").mean())
        .sort_values(ascending=False)
    )

    print("\nResignation Rate Per Department\n")
    print(resignation_rate)

    top_10_employees = df.nlargest(10, "performance_score")

    print("\nTop 10 Employees\n")
    print(top_10_employees[["name", "department", "performance_score"]])

    high_performance = df[
        (df["performance_score"] > 90)
        & (df["attendance_rate"] > 95)
    ]

    print("\nHigh Performance Employees\n")
    print(high_performance[["name", "performance_score", "attendance_rate"]])

    df["remote_work_numeric"] = (
        df["remote_work"]
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    overtime_corr = df["overtime_hours"].corr(df["performance_score"])
    training_corr = df["training_hours"].corr(df["performance_score"])
    remote_corr = df["remote_work_numeric"].corr(df["performance_score"])
    salary_corr = df["salary"].corr(df["experience_years"])
    work_hours_corr = df["work_hours_per_week"].corr(
        df["performance_score"]
    )

    def evaluate_correlation(value, feature_1, feature_2):
        if abs(value) >= 0.7:
            strength = "Very Strong"
        elif abs(value) >= 0.5:
            strength = "Strong"
        elif abs(value) >= 0.3:
            strength = "Moderate"
        elif abs(value) >= 0.1:
            strength = "Weak"
        else:
            strength = "No Clear"

        if value > 0.1:
            direction = "Positive"
        elif value < -0.1:
            direction = "Negative"
        else:
            direction = "Neutral"

        print(
            f"{feature_1} has a {direction} "
            f"{strength} correlation with "
            f"{feature_2} (r = {value:.2f})"
        )

    print("\nCorrelation Analysis\n")

    evaluate_correlation(
        overtime_corr,
        "Overtime Hours",
        "Performance Score"
    )

    evaluate_correlation(
        training_corr,
        "Training Hours",
        "Performance Score"
    )

    evaluate_correlation(
        remote_corr,
        "Remote Work",
        "Performance Score"
    )

    evaluate_correlation(
        salary_corr,
        "Salary",
        "Experience Years"
    )

    evaluate_correlation(
        work_hours_corr,
        "Work Hours Per Week",
        "Performance Score"
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(df["salary"], bins=10, kde=True)

    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Count")

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["age"])

    plt.title("Age Distribution")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="overtime_hours",
        y="performance_score",
        hue="department",
        s=100
    )

    plt.title("Overtime Hours vs Performance")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="training_hours",
        y="performance_score",
        hue="department",
        s=100
    )

    plt.title("Training Hours vs Performance")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="department")

    plt.title("Employees Per Department")
    plt.xlabel("Department")
    plt.ylabel("Employees Count")

    plt.xticks(rotation=20)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x="department",
        y="performance_score"
    )

    plt.title("Average Performance by Department")
    plt.xlabel("Department")
    plt.ylabel("Performance Score")

    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("employees.csv file was not found.")

except Exception as error:
    print(error)

else:
    print("\nDataset processed successfully.")
