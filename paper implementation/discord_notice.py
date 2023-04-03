from urllib import request
from xxlimited import foo
from discord_webhook import DiscordWebhook, DiscordEmbed
from time import sleep
from datetime import datetime

webhook_url = "https://discord.com/api/webhooks/1045216047462092830/b1hDz1laD2O7zI50dJtQ6GGG7H2cRbUB7PSV3RBcFtA-7FWEOJ6uYJfJGv-g9wbwV6yR"

def send_message(title, description):
    webhook = DiscordWebhook(url=webhook_url)
    
    embed = DiscordEmbed(title=title, description=description, color='03b2f8')
    embed.set_author(name='status', icon_url='https://cdn-icons-png.flaticon.com/128/2291/2291988.png')
    embed.set_timestamp()

    webhook.add_embed(embed)
    status = webhook.execute()

def start():
    now = datetime.now()
    msg = now.strftime('%Y-%m-%d %H:%M:%S')
    send_message(title='is running', description=f'{msg}')
    return

def end():
    now = datetime.now()
    msg = now.strftime('%Y-%m-%d %H:%M:%S')
    send_message(title='is done', description=f'{msg}')
    return