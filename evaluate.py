from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def evaluate_answer(predicted, reference):
    predicted = clean_text(predicted)
    reference = clean_text(reference)

    # BLEU Score
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([reference.split()], predicted.split(), smoothing_function=smooth)

    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, predicted)

    # Exact Match
    exact = int(predicted == reference)

    return {
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(rouge['rouge1'].fmeasure, 3),
        "ROUGE-L": round(rouge['rougeL'].fmeasure, 3),
        "Exact Match": exact
    }
