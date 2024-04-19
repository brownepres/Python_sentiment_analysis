- [] Descriptive README file
- [x] Create test_branch
- [] Organize folders, put functions to different folder
- [] Unit test

This is a personal project by Barnabás Epres to showcase the differences between the two sentiment analyser models, VADER and roBERTa, and display several inferential statistic values. 
The aim of this project was to demonstrate how the two models differentiate from each other. VADER model uses a method to determine the sentiment for each word in a sentence, then calculates the weighted average value to make an overall compound score for the input text's sentiment. The roBERTa model works on similar basis, however it takes the context and surrounding words into account, therefore it gives a smoother approach to sentences. My goal was to demostrate this difference, especially while analysing edge-cases (e.g.: sarcasm, irony), and to show that VADER model misunderstands such situations, while roBERTa recognises them in most cases. 

Difficulties: the VADER model calculates positiv, negativ and neutral sentiment value alongside with a combined so-called compound value, on a scale from -1 to 1, -1 being negative sentiment. On the other hand roBERTa model operates with only the three sentiment values (pos, neu, neg), all of them calulated on a scale from 0 to 1, 0 being negative sentiment. To overcome the challenges caused by the used measurements, I normalized the three output values of the roBERTa model and applied a hyperbolic tangent function on them to calculate roBERTa compound. With this modification, both models had a compound value measured on the same scale, from -1 to 1. The idea of the hyperbolic tangent function came from Linda Erwe's and Xin Wang's [paper](https://lup.lub.lu.se/student-papers/search/publication/9145112) on the matter.
