from convokit import Corpus

corpus = Corpus(filename="/home/vm-user/Desktop/connl/data/switchboard-processed-corpus")

# Collect metadata keys
keys = set()
for utt in corpus.iter_utterances():
    keys.update(utt.meta.keys())

# Print anything that looks like audio or alignment info
cands = [k for k in sorted(keys) if any(s in k.lower() for s in ["audio", "wav", "mp3", "path", "file", "start", "end", "time", "offset"])]
print("Candidate keys:", cands)

# Show one utterance meta so we can see what fields exist
u = next(corpus.iter_utterances())
print("\nExample utt_id:", u.id)
print("Example text:", u.text)
print("Example meta:", u.meta)
