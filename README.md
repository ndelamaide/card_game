# Battle card game

The goal was to automatically count the points of 4 players in a battle card game. The data available to us were the pictures of the table after each round, like below.

![example_round](https://user-images.githubusercontent.com/44365345/155527516-af29bcfa-221f-4932-b9d0-f606f5440793.jpg)

The steps we took to solve the problem are the following:

- Pre-processing : contrast stetching + multi-otsu thresholding
- Find the bounding boxes of the cards and dealer token
- Extract the cards
- Detect the rank and suite of each card
- Create a dataset using the extracted ranks and suites of each game
- Train an MLP classifier to identify the ranks and suites
- Create a pipeline to count the points of each player
