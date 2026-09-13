Titanic Survival Prediction

Predicting whether a passenger survived the Titanic using decision trees and random forests, and figuring out which features actually matter versus which just look important.

Data: Kaggle's Titanic dataset (train.csv), 891 rows, 12 columns.

Cleaning and features:

Embarked had 2 missing values, filled with the mode and one-hot encoded. Age was missing for 177 passengers, so I filled those with the median age within each passenger class rather than the overall median, since age varies a lot by class. Cabin was missing for 687 of 891 rows, way too many to use directly, so instead of the actual cabin values I made a Has_Cabin flag. Sex got a simple binary encoding. I also pulled titles out of the Name column (Mr, Mrs, Miss, Master), lumped the 13 rare ones (Dr, Rev, Col, etc.) into a single Rare bucket, and one-hot encoded that too.

Final features: Pclass, sex_encoded, Age, Fare, Has_Cabin, Title_Miss, Title_Mr, Title_Mrs, Title_Rare.

Results (random_state=42):

| Model | Train | Test |
|-------|-------|------|
| Decision tree, depth 5 | 84.7% | 83.8% |
| Decision tree, unlimited depth | 97.8%+ | ~76% |
| Random forest, 100 trees, depth 10 | 94.0% | 84.4% |

Random forest confusion matrix: 96 correctly predicted deaths, 55 correctly predicted survivors, 9 false positives, 19 false negatives. Precision/recall came out to 0.83/0.91 for "died" and 0.86/0.74 for "survived."

What I actually learned doing this:

An unlimited-depth tree memorizes the training set 97.8% train accuracy but only 76% on test data. Capping max_depth at 5 brought test accuracy up to 83.8% even though train accuracy dropped. Less "confident" on paper, more correct in practice.

Random forest beat the single tree by a couple points, which tracks with how it works — 100 trees, each trained on a different random slice of rows and features, so their individual mistakes don't overlap but the real patterns (sex, class, fare) show up in almost all of them and survive the vote.

More features isn't automatically better. Adding SibSp, Parch, and Embarked actually dropped test accuracy slightly their importance scores were tiny, and they mostly just gave the tree extra ways to fit noise.

Feature engineering beat throwing in raw columns. Pulling Title out of Name turned out to matter more than I expected: adult men (Mr) survived at 15.7%, but boys (Master) survived at 57.5%, a gap that plain Sex can't see since both are just "male."

Feature importance isn't the same as "how much a feature matters." When I dropped Fare, the freed-up importance mostly went to Age, not Pclass, even though Pclass is more obviously related to Fare. Continuous features like Age get more chances to be used in splits than a 3-value column like Pclass, so they tend to look more important regardless of how predictive they really are.

The model is worse at spotting survivors than deaths, recall was 91% for deaths versus 74% for survivors. There were more deaths than survivors in the training data, so when the model's unsure, it leans toward the more common outcome.
