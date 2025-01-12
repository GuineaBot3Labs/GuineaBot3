![GuineaBot4](/730b2f04-e42c-450f-90c2-51ea20b5b272.jpg)

  [Report bug][issue-link]
  · [wiki][wiki-link]
  · [Lichess API Usage][API-link]
  · [Contribute][contribution-link]

  Copyright © 2022 GuineaPigLord. All rights reserved.
  
# GuineaBot3

A lichess bot that self learns and improves over time. You can watch GuineaBot3's games <a href="https://lichess.org/@/GuineaBot3/tv">here</a>

### WARNING: THIS CODE NEEDS A SUPER HIGH END GPU WHICH MAY NOT BE ACCESSIBLE FOR EVERYONE, HERE IS A MORE [COMPACT VERSION][compact-link], HOWEVER, FOR THIS BRANCH YOU WILL NEED A CUDA COMPATIBLE GPU WITH A MEMORY OF 24GBs. I USED A TESLA K80 FOR THIS, YOU WILL NEED TO INSTALL TORCH MANUALLY IF YOU WANT TO DO THIS WITH THE SAME GPU. ###

## UPDATES ##
In GuineaBot v4.1.9, we have added self play capability and pgn replay, you can vary how many games to self play within the replay_from_pgn() function, we have found that tinkering with the settings is the best approach, just like 3d printing, this model needs to be fine tuned. We have also found that pgn files should be under 10,000 games unless you want a super high end grandmaster that takes weeks or even months to train. Please be cautious as this project can severely add to your electricity bill.

## FUTURE UPDATES ##
In GuineaBot v4.2, we are making major changes, which will make GuineaBot much more compact, and will no longer need alternate versions for singular and multi-GPU performance, and it will hopefully work on 1 GB of memory. The following update will also introduce a GAN (Generative Adversarial Network,) to predict opponent actions and "look-ahead" into the future. Additionally, GuineaBot will no longer have a static reward function as rewards will be determined via an actor-critic policy.

## SETUP ##

You can set up this bot easily, it requires a powerful GPU compatible with CUDA, you can vary how many GPUs you want to use.

You will need to have python3.10 or python3.11 for this code, you can download it as shown down below:

### LINUX ###

    sudo apt install python3
    sudo apt install git
    git clone https://github.com/GuineaBot3Labs/GuineaBot3.git GuineaBot3
    cd GuineaBot3

NOTE: You will have to compile python3.10 from source, this is more difficult but can be rewarding. Follow the last 3 steps after installing from source.

### WINDOWS ###
for windows go to this <a href="https://python.org">website</a> and download the version you want, then download the zip on this page and extract it to your prefered location.

    cd GuineaBot3


### FINAL STEPS ###

sign up for lichess.org and upgrade your account to a bot account, then use pip and do the following command:

for linux:

    python3 -m pip install -r requirements.txt

for windows:

    py -m pip install -r requirements.txt

finally you will have to have cuda installed. this is the hardest part because you need a GPU with 24 GBs of memory AND cuda compatible, plus, not every GPU is created equal and you will have to get the version of cuda your GPU is compatible with, pytorch also complains about incompatible versions and you will have to get your hands dirty. If you don't have this luxury you can always play against my bot <a href="https://lichess.org/@/GuineaBot3">here</a> or maybe try out my more compact bot [here.][compact-link]

## USAGE ##

To use this program simply type in your 3 terminal windows (for linux):

    python3 GuineaBot3.py

    python3 waitforrequests.py

    python3 challenge_bot.py

and on windows on your 3 command prompts type:

    py GuineaBot3.py

    py waitforrequests.py

    py challenge_bot.py

and just like that you are up and running, WHEEK WHEEK!!!

### Contribution ###

You can contribute [here.][contribution-link] Fork this repository and make your modifications, be it compact, Crazyhouse, etc.

[issue-link]: ../../issues/new
[wiki-link]: ../../wiki
[API-link]: https://lichess.org/api#tag/Bot
[contribution-link]: ../../fork
[compact-link]: ../../tree/compact
