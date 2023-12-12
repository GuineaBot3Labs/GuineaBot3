![GuineaBot4](/730b2f04-e42c-450f-90c2-51ea20b5b272.jpg)

  [Report bug][issue-link]
  · [wiki][wiki-link]
  · [Lichess API Usage][API-link]
  · [Contribute][contribution-link]

  Copyright © 2022 GuineaPigLord. All rights reserved.
  
# GuineaBot3

A lichess bot that self learns and improves over time. You can watch GuineaBot3's games <a href="https://lichess.org/@/GuineaBot3/tv">here</a>

### COMPACT EDITION, WHEEK WHEEK!!! ###

## UPDATES ##
In GuineaBot3 v4.1.9, we have added self play capability and pgn replay, you can vary how many games to self play within the replay_from_pgn() function, we have found that tinkering with the settings is the best approach, just like 3d printing, this model needs to be fine tuned. We have also found that pgn files should be under 10,000 games unless you want a super high end grandmaster that takes weeks or even months to train. This edition does not add much to electricity bill, it can run 4 instances on a single 12 GB GPU, adding that up it is only 4 GB allocation, very suitable for a cpu even, WHEEK WHEEK!!!

## SETUP ##

You can set up this bot very easily, it requires a very good GPU cuda compadible and multicore, you can vary how many cores you want to use.

You will need to have python3.10 or python3.11 for this code, you can download it as shown down below:

### LINUX ###

    sudo apt install python3
    sudo apt install git
    git clone https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot.git GuineaBot3
    cd GuineaBot3

NOTE: You will have to do python3.10 from source, this is more difficult but can be rewarding. Follow the last 3 steps after installing from source.

### WINDOWS ###
for windows go to this <a href="https://python.org">website</a> and download the version you want, then download the zip on this page and extract it to your prefered location.

    cd deep-GuineaBot3-lichess-bot


### FINAL STEPS ###

sign up for lichess.org and upgrade your account to a bot account, then use pip and do the following command:

for linux:

    python3 -m pip install -r requirements.txt

for windows:

    py -m pip install -r requirements.txt

finally you will have to have cuda installed. this is the hardest part because you need a GPU with 24 GBs of memory AND cuda compatible. If you don't have this luxury you can always play against my bot <a href="https://lichess.org/@/GuineaBot3">here</a> or maybe try to make my model more lightweight, I would appreciate the help because I am currently working on this same part.

## USAGE ##

To use this program simply type in your 3 terminal windows (for linux):

    python3 GuineaBot3.py

    python3 waitforrequests.py

    python3 challenge_bot.py

and on windows on your 3 command prompts type:

    py GuineaBot3.py

    py waitforrequests.py

    py challenge_bot.py
### Contribution ###

You do not need any pull requests here, however please be cautious, some could probably do damage to your computer but do not let that discourage you, we want to push out those black hat hackers and keep this lab safe, you can contribute [here.][contribution-link] Click on the "New branch" button to make a new branch, be it compact, CPU compatible, arm robot, failure is only another achievement! Without failure, we wouldn't have learned coding, now let's learn AI!

[issue-link]: https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot/issues/new
[wiki-link]: https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot/wiki
[API-link]: https://lichess.org/api#tag/Bot
[contribution-link]: https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot/branches
