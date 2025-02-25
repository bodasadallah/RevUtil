import spacy
from wtpsplit import SaT
import torch
from utils import merge_short_sentences
device = "cuda" if torch.cuda.is_available() else "cpu"
sat = SaT("sat-3l")
nlp = spacy.load("en_core_web_sm")
sat.half().to(device)
def split_sentences(text: str) -> list[str]:
  doc = nlp(text)
  sentences = [sent.text for sent in doc.sents]
  return sentences

def group_sentences_semantically(sentences: list[str], threshold: int) -> list[str]:
  docs = [nlp(sentence) for sentence in sentences]
  segments = []
  start_idx = 0
  end_idx = 1
  segment = [sentences[start_idx]]
  while end_idx < len(docs):
    if docs[start_idx].similarity(docs[end_idx]) >= threshold:
      segment.append(str(docs[end_idx]))
    else:
      segments.append(" ".join(segment))
      start_idx = end_idx
      segment = [sentences[start_idx]]
    end_idx += 1

  if segment:
    segments.append(" ".join(segment))

  return segments

def split_paragraph_semantically(text: str, threshold= 0.6) -> list[str]:
#   sentences = split_sentences(text)
    sentences = sat.split(text)
    grouped_sentences = group_sentences_semantically(sentences, threshold)
    return merge_short_sentences(grouped_sentences)


