![GuineaBot4](/730b2f04-e42c-450f-90c2-51ea20b5b272.jpg)

  [Report bug][issue-link]
  · [wiki][wiki-link]
  · [Lichess API Usage][API-link]
  · [Contribute][contribution-link]

  Copyright © 2022 GuineaPigLord. All rights reserved.
  
# GuineaBot3-Experimental branch

### WARNING: This branch is meant to be a branch for edits that may or may not cause errors and other complications. This is meant to test GuineaBot3 and vet it of any potential bugs/glitches. ###

## SETUP ##

You can set up this bot very easily, it requires a very good GPU cuda compadible and multicore, you can vary how many cores you want to use.

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
