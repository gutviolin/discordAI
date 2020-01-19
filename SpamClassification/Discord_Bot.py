import discord
import spam_classification_ASCII as classifier

token = "DISCORD_BOT_TOKEN"

client = discord.Client()
messages = []

@client.event
async def on_message(message):
    message_content = message.content

    if message.author.bot == False:
        messages.append(message.content)
        if len(messages)  >= 3:
            messages.pop(0)
        if (message.content[:5]) == "=spam" or (message.content[:8]) == "=notspam" or  (message.content[:6]) == "=train":
            if "Spam Lord" in [role.name for role in message.author.roles]:
                if (message.content[:1]) == "=":

                    if (message.content[1:]) == "spam":
                        with open("Spam.txt", "a+") as text_file:
                            await message.channel.send("Added '" +messages[len(messages)-2] +"' to spam dataset!")
                            text_file.write(messages[len(messages)-2] +"\n")

                    if (message.content[1:]) == "notspam":
                        with open("NotSpam.txt", "a+") as text_file:
                            await message.channel.send("Added '" +messages[len(messages)-2] +"' to not-spam dataset!")
                            text_file.write(messages[len(messages)-2] +"\n")

                if (message_content) == "=train":
                    await message.channel.send("Training model with updated dataset...")
                    classifier.train_model()
                    await message.channel.send("Done!")

                if (message_content[:6]) == "=spam ":
                        with open("Spam.txt", "a+") as text_file:
                            await message.channel.send("Added '" +message_content[6:] +"' to spam dataset!")
                            text_file.write(message_content[6:] +"\n")

                if (message_content[:9]) == "=notspam ":
                    print(message_content[9:])
                    with open("NotSpam.txt", "a+") as text_file:
                        await message.channel.send("Added '" +message_content[9:] +"' to not-spam dataset!")
                        text_file.write(message_content[9:] +"\n")

            else:
                await message.channel.send("You do not have permission!")

        if (message.content[:1]) == "=":
                if (message.content[1:]) == "ben":
                    await message.channel.send("https://pbs.twimg.com/profile_images/1084873514291585024/7eWpkx-c_400x400.jpg")

        if (message.content[:8]) != "=notspam" and (message.content[:5]) != "=spam" and (message.content) != "=train":
            if classifier.test_data_msg(message.content) == "Spam":
                #print(message.content)
                await message.channel.send("I think this message is spam...")

client.run(token)
