import asyncio
import discord
from src.response import ResponseGenerator
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

intents = discord.Intents(messages=True, guilds=True, typing = False, presences = False, members=False)
client = discord.AutoShardedClient(intents=intents, chunk_guilds_at_startup=False)
config_general=json.loads(open("config-bot.json","r").read())

model=ResponseGenerator("https://woz-model.herokuapp.com/v1/models/jade:predict")

@client.event
async def on_ready():
    print('Logged in as '+client.user.name+' (ID:'+str(client.user.id)+') | Connected to '+str(len(client.guilds))+' servers')
    print('--------')
    print("Discord.py verison: " + discord.__version__)
    print('--------')
    print(str(len(client.shards))+" shard(s)")
    
enabled_guilds=[387328200113651732,669808453497258024, 692688645559418930]
    
@client.event
async def on_message(message):
    loop = asyncio.get_event_loop()
    if message.author.bot == False and message.guild != None:
        if message.content.lower().startswith(config_general["prefix"]):
            if message.guild.id in enabled_guilds:
                message.content=message.content[len(config_general["prefix"]):]
                if message.content == "-h":
                    logs=model.history[message.author.id]
                    await message.reply("History:\n> "+"\n> ".join(logs["history"])+f"\nLast seen: {datetime.fromtimestamp(logs['timestamp'])}")
                if message.content == "-r":
                    model.reset(message.author.id)
                    await message.reply("Successfully reset your history with Jade")
                else:
                    async with message.channel.typing():
                        out = await loop.run_in_executor(ThreadPoolExecutor(), model.response, message.author, message.content, True)
                        await message.reply(out.replace("@everyone","").replace("@here", ""))
            else:
                await message.reply("This sever does not have the test model enabled. This process must be done manually by Jess- Join the support server here: https://discord.com/invite/eGffUFY")

client.run(config_general["token"])