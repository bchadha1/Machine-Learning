README (Machine Leaning Assignment 1)

InfogainHeurML.py can be executed from command-line or IDE
The program will take inputs from the user for the values of n and K and the paths of the directories where the DATASETS are stored.
The output will be
<Two decision Trees> - D1 implements Information gain heuristic - D2 implements Variance impurity heuristic
The program also calculates the accuracies of the two Heuristic methods before and after Decision Tree Prunning

INPUT:
Enter the inputs with spaces: <No. of iterations post-prunning (eg. 10, 15, etc)> <Upper limit of prunning factor (eg. 10, 15, etc)> <Training_Set directory> <Validation_Set directory> <Test_Set directory> <Print Tree?Yes/No>
Example ->> 3 5 C:\Users\bhuva\OneDrive\Desktop\ML_Assignment\dataset1\training_set.csv C:\Users\bhuva\OneDrive\Desktop\ML_Assignment\dataset1\validation_set.csv C:\Users\bhuva\OneDrive\Desktop\ML_Assignment\dataset1\test_set.csv yes

OUTPUT:
Decision tree D1 (Information Gain Heuristic):
PI= 0 :
| PH= 0 :0
| PH= 1 :
| | PB= 0 :0
| | PB= 1 :
| | | PR= 0 :0
| | | PR= 1 :
| | | | PK= 0 :0
| | | | PK= 1 :
| | | | | PC= 0 :1
| | | | | PC= 1 :0
PI= 1 :
| PH= 0 :
| | Pn= 0 :0
| | Pn= 1 :
| | | PV= 0 :0
| | | PV= 1 :
| | | | PG= 0 :
| | | | | PK= 0 :1
| | | | | PK= 1 :0
| | | | PG= 1 :0
| PH= 1 :
| | PC= 0 :
| | | PR= 0 :
| | | | PD= 0 :
| | | | | Pn= 0 :
| | | | | | Pa= 0 :
| | | | | | | PK= 0 :0
| | | | | | | PK= 1 :1
| | | | | | Pa= 1 :1
| | | | | Pn= 1 :0
| | | | PD= 1 :
| | | | | Pa= 0 :
| | | | | | PG= 0 :0
| | | | | | PG= 1 :1
| | | | | Pa= 1 :0
| | | PR= 1 :
| | | | PB= 0 :
| | | | | PL= 0 :0
| | | | | PL= 1 :
| | | | | | Po= 0 :
| | | | | | | PJ= 0 :1
| | | | | | | PJ= 1 :
| | | | | | | | PF= 0 :0
| | | | | | | | PF= 1 :1
| | | | | | Po= 1 :
| | | | | | | PM= 0 :0
| | | | | | | PM= 1 :
| | | | | | | | PE= 0 :1
| | | | | | | | PE= 1 :0
| | | | PB= 1 :
| | | | | Pd= 0 :1
| | | | | Pd= 1 :
| | | | | | PF= 0 :0
| | | | | | PF= 1 :
| | | | | | | PD= 0 :
| | | | | | | | PM= 0 :0
| | | | | | | | PM= 1 :1
| | | | | | | PD= 1 :1
| | PC= 1 :
| | | PB= 0 :
| | | | PG= 0 :
| | | | | Pa= 0 :
| | | | | | PK= 0 :1
| | | | | | PK= 1 :0
| | | | | Pa= 1 :0
| | | | PG= 1 :0
| | | PB= 1 :0

Decision tree D2 (Variance Impurity Heuristic):
PI= 0 :
| PH= 0 :0
| PH= 1 :
| | PB= 0 :0
| | PB= 1 :
| | | PR= 0 :0
| | | PR= 1 :
| | | | PK= 0 :0
| | | | PK= 1 :
| | | | | PC= 0 :1
| | | | | PC= 1 :0
PI= 1 :
| PH= 0 :
| | Pn= 0 :0
| | Pn= 1 :
| | | PV= 0 :0
| | | PV= 1 :
| | | | PG= 0 :
| | | | | PK= 0 :1
| | | | | PK= 1 :0
| | | | PG= 1 :0
| PH= 1 :
| | PC= 0 :
| | | PR= 0 :
| | | | PD= 0 :
| | | | | Pn= 0 :
| | | | | | Pa= 0 :
| | | | | | | PK= 0 :0
| | | | | | | PK= 1 :1
| | | | | | Pa= 1 :1
| | | | | Pn= 1 :0
| | | | PD= 1 :
| | | | | Pa= 0 :
| | | | | | PG= 0 :0
| | | | | | PG= 1 :1
| | | | | Pa= 1 :0
| | | PR= 1 :
| | | | PB= 0 :
| | | | | PL= 0 :0
| | | | | PL= 1 :
| | | | | | Po= 0 :
| | | | | | | PJ= 0 :1
| | | | | | | PJ= 1 :
| | | | | | | | PF= 0 :0
| | | | | | | | PF= 1 :1
| | | | | | Po= 1 :
| | | | | | | PM= 0 :0
| | | | | | | PM= 1 :
| | | | | | | | PE= 0 :1
| | | | | | | | PE= 1 :0
| | | | PB= 1 :
| | | | | Pd= 0 :1
| | | | | Pd= 1 :
| | | | | | PF= 0 :0
| | | | | | PF= 1 :
| | | | | | | PD= 0 :
| | | | | | | | PM= 0 :0
| | | | | | | | PM= 1 :1
| | | | | | | PD= 1 :1
| | PC= 1 :
| | | PB= 0 :
| | | | PG= 0 :
| | | | | Pa= 0 :
| | | | | | PK= 0 :1
| | | | | | PK= 1 :0
| | | | | Pa= 1 :0
| | | | PG= 1 :0
| | | PB= 1 :0

Process finished with exit code 1

After implementing the two heuristics, applies the post-prunning algorithm to reduce overfitting.