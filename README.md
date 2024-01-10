![GuineaBot4](/730b2f04-e42c-450f-90c2-51ea20b5b272.jpg)

  [Report bug][issue-link]
  · [wiki][wiki-link]
  · [Lichess API Usage][API-link]
  · [Contribute][contribution-link]

  Copyright © 2022 GuineaPigLord. All rights reserved.
  
# GuineaBot3

A lichess bot that self learns and improves over time. You can watch GuineaBot3COMPACT's games <a href="https://lichess.org/@/GuineaBot3COMPACT/tv">here</a>

## UPDATES ##
### PARALLEL-COMPACT EDITION, WHEEK WHEEK!!! ###
In GuineaBot3 v4.2.0, PARALLEL-COMPACT EDITION, in this version, we have added parallel computing to the model, this can be useful if you have two 2 GB GPUs, or you simply want training time speed. This edition does not add much to electricity bill, it can run up to 8 instances on two 12 GB GPUs, WHEEK WHEEK!!!

## SETUP ##

You can set up this bot very easily, just like it's big brother, however unlike it's big brother, it does not requires a very good GPU that is cuda compadible and multicore, if you want to use multiple cores or want to use a really big GPU, see it's big brother [here.](https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot)

You will need to have python3.10 or python3.11 for this code, you can download it as shown down below:

### LINUX ###

    sudo apt install python3
    sudo apt install git
    git clone -b compact https://github.com/ethicalhacker7192/deep-GuineaBot3-lichess-bot.git GuineaBot3
    cd GuineaBot3

NOTE: You will have to do python3.10 from source, this is more difficult but can be rewarding. Follow the last 3 steps after installing from source.

### WINDOWS ###
for windows go to this <a href="https://python.org">website</a> and download the version you want, then download the zip on this page and extract it to your prefered location.

    cd deep-GuineaBot3-lichess-bot


### FINAL STEPS ###

sign up for lichess.org and upgrade your account to a bot account see the API usage [here][API-link], then use pip and do the following command:

for linux:

    python3 -m pip install -r requirements.txt

for windows:

    py -m pip install -r requirements.txt

finally, you can skip this part as this part is optional, if you do not have cuda installed and you want to run this on a GPU for training acceleration, you can download cuda on the official website along with the needed nvidia driver, this part is a pain, if you just want to see the thing working, you can always play against my bot <a href="https://lichess.org/@/GuineaBot3COMPACT">here</a>, of course this is the COMPACT bot, not the Parallel-COMPACT bot, so I will make a new account for the parallel-compact version soon! WHEEK WHEEK!!!

## USAGE ##

To use this program simply type in your 3 terminal windows (for linux):

    python3 GuineaBot3.py

    python3 waitforrequests.py

    python3 challenge_bot.py

and on windows on your 3 command prompts type:

    py GuineaBot3.py

    py waitforrequests.py

    py challenge_bot.py
## Contribution ##

You do not need any pull requests here, however please be cautious, some could probably do damage to your computer but do not let that discourage you, we want to push out those black hat hackers and keep this lab safe, you can contribute [here.][contribution-link] Feel free to fork as much as you want, just follow the rules of conduct, be it chaotic, broken, exploded, failure is only another achievement! Without failure, we wouldn't have learned coding, now let's learn AI, WHEEK WHEEK!!!

[issue-link]: ../../issues/new
[wiki-link]: ../../wiki
[API-link]: https://lichess.org/api#tag/Bot
[contribution-link]: ../../fork
[compact-link]: ../../tree/compact
