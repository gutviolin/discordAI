# discordAI
Discord neural network spam filter bot

A discord bot that uses a python based neural network to classify "spam" messages from messages that look like english text.
Built with tensorflow and the discord python API, uses a dataset made very hastily so the outcome was not so accurate. 
Dataset is based on movie reviews (for the dataset of not spam text) and completely randomly genereated ASCII characters
for the spam dataset so does not refelct the classification of spam accurately.
The implemenation will require you to add your bot token to the Discord_Bot.py file to work and also allows users with a role called
"Spam Lord" to label and message sent in the server as spam or notspam using the commands =spam =notspam respectively. This will add
the message to the dataset in an attempt to make the dataset closer to real world data.
This was my fist neural network project and really my first major coding project so any edits or suggestions are much appreciated, thanks!
 Require tensorflow and numpy
