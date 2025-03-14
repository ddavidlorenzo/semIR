from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """When the future began...The men had it. Yeager. Conrad. Grissom. Glenn. Heroes... the first Americans in space... battling the Russians for control of the heavens putting their lives on the line. The women had it. While Mr. Wonderful was aloft, it tore your heart out that the Hero's Wife, down on the ground, had to perform with the whole world watching... the TV Press Conference: "What's in your heart? Do you feel with him while he's in orbit?" The Right Stuff. It's the quality beyond bravery, beyond courage. It's men like Chuck Yeager, the greatest test pilot of all and the fastest man on earth. Pete Conrad, who almost laughed himself out of the running. Gus Grissom, who almost lost it when his capsule sank. John Glenn, the only space traveller whose apple-pie image wasn't a lie.
"""

output = summarizer(ARTICLE, max_length=128, min_length=30, do_sample=False)
print(output)