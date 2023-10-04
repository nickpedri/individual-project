# Project goals

* Find online data for the project
# Prepare, explore data
# Create models to predict the target variable
# Analyze results and think about the results
# Reflect on possible improvements for next time and lessons learned


# Project description:

For this project I acquired my data from data world. Here is the link to the data: https://data.world/shruthi12/animal-planet/workspace/file?filename=AustinAnimalWorld.csv. The data was simply read into python via pandas read csv function.
Once acquired, a lot of cleaning and preparing was done. This involved binning data, dropping columns, splitting columns into
cleaner more organized columns. I then explored my data to find useful information for modeling. During modeling the data was split into
4 groups one for each type of animal. So the final result was 4 models one for each type of animal. 


# Project plan:

* Create a rough draft (plan) for project.
* Find data online, preferably API, to create a project with.
* Acquire data
* Prepare data for exploration
* Explore data. Annotate key findings.
* Pre-process data for modeling
* Create models and fine tune models
* Pick best model(s) and form conclusions


# Project questions

* Do different animals have different ratios of outcomes? e.g. Do dogs get adopted more often than cats?
* Does the age of an animal affect the outcome? 
* Does the breed of an animal affect its outcome?
* Does the gender of an animal afect its outcome?


# Data dictionary
 
| Feature      |              Description                   |
| --------     |               -------                      |
|name          |   True or false if animal has name         |
|outcome_type  |  The outcome of what happens to the animal |
|animal_type   |   Type of animal                           |
|color         |         Color of the animal.               |
|age           |           age of the animal in days.       |
|gender        |            Gender of the animal.           |
|neut_spay     | T or F for if animal is neutered or spayed |
|condition     |       Condition of the animal.             |
|breed1        |     Breed of the animal                    |
|breed2        |           Second breed of animal           |


# Instructions

1. Acquire data from https://data.world/shruthi12/animal-planet/workspace/file?filename=AustinAnimalWorld.csv.

2. Download and install Anaconda, and install Python through Anaconda so that you have all of the necessary data science libraries and tools that you will need for pyhton.

3. Once you have Python, all of the necessary libraries, and the .csv file containing the animals data,  you are all set to go.


# Key findings:

1. Animals with no name are less likely to be adopted or returned to their owner. They most likely do not have owners.
0. Animals of unknown gender are way more likely to be euthanized or transfered than adopted.
0. Animals between 0-2 years old are adopted at much higher rates than older ones.
0. 'Other' animals are euthanized at way higher rates. This is likely because they are wild animals.
0. Older animals are more likely to be returned to their owners

Next time, given more time. I would like to explore data more and prepare it differently. I would also like more time test different hyperparameters on models. The models created do perform much better than the baseline. It is hard to draw a clear conclusion on how each breed of animal impacts their outcome, but from looking at the visuals it is clear that different breeds do have different outcomes. I would also next time create a simpler mvp and then improve upon that to create a better model.