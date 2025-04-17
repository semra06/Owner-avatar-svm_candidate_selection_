“Candidate Selection in Recruitment:

Project Title: “Candidate Selection in Recruitment: Application Evaluation with SVM”

Scenario: Scenario: You are a human resources team member at a technology company. Your goal is to develop a model that predicts whether candidates applying for a software developer position will be hired based on their years of experience and technical exam score.

Data Features: For each application, you have the following information:

year_of_experience: Total software experience of the candidate (0–10 years)

technical_point: Technical exam score (0–100)

label:

1: Not hired (unsuccessful candidate)

0: Hired (successful candidate)

Labeling Criteria (rule-based):

Those with less than 2 years of experience and exam score below 60 are not hired.

Tasks: At least 200 applications were generated with random.

Labeled with the rule above based on experience and technical score.

Data was separated into training and testing. Data was scaled with StandardScaler. The model was trained with SVC(kernel='linear'). The decision boundary was visualized with matplotlib. The prediction was made by getting experience and technical points from the user. The success was evaluated with accuracy_score, confusion_matrix, classification_report.

Development Areas: Trying non-linear classes by changing the kernel Converting the model to service with FastAPI Parameter tuning (C, gamma) (R&D) 
