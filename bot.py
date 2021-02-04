import asyncio
import discord
from interact_web import ResponseGenerator
import json
from concurrent.futures import ThreadPoolExecutor

intents = discord.Intents(messages=True, guilds=True, typing = False, presences = False, members=False)
client = discord.AutoShardedClient(intents=intents, chunk_guilds_at_startup=False)
config_general=json.loads(open("config-bot.json","r").read())["discord"]

model=ResponseGenerator("https://woz-model.herokuapp.com/v1/models/jade:predict")

@client.event
async def on_ready():
    print('Logged in as '+client.user.name+' (ID:'+str(client.user.id)+') | Connected to '+str(len(client.guilds))+' servers')
    print('--------')
    print("Discord.py verison: " + discord.__version__)
    print('--------')
    print(str(len(client.shards))+" shard(s)")
    
@client.event
async def on_message(message):
    loop = asyncio.get_event_loop()
    if message.author.bot == False and message.guild != None:
        if message.content.lower().startswith(config_general["prefix"]):
            if message.guild.id == 387328200113651732:
                message.content=message.content[len(config_general["prefix"]):]
                if message.content == "-h":
                    message.reply("History:\n>"+"\n>".join(model.history))
                else:
                    await message.channel.trigger_typing()
                    out = await loop.run_in_executor(ThreadPoolExecutor(), model.response, message.author.display_name, message.content, True)
                    await message.reply(out).replace("@everyone","").replace("@here", "")

client.run(config_general["token"])