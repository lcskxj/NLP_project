from textstat import textstat as tst

MEAN_GRADE = 10

MEAN_SL = 23.539
STD_SL = 6.401

MEAN_AL = 5.617
STD_AL = 0.397

MEAN_SC = 6.06
STD_SC = 2.453

def get_score(text):
    scores = []
    scores.append((tst.avg_sentence_length(text) - MEAN_SL) / STD_SL)
    scores.append((tst.avg_letter_per_word(text) - MEAN_AL) / STD_AL)
    scores.append(tst.avg_sentence_per_word(text))
    scores.append((tst.sentence_count(text) - MEAN_SC) / STD_SC)
    scores.append((tst.flesch_kincaid_grade(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.flesch_reading_ease(text) - 50) / 50)
    scores.append((tst.smog_index(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.coleman_liau_index(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.automated_readability_index(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.dale_chall_readability_score(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.linsear_write_formula(text) - MEAN_GRADE) / MEAN_GRADE)
    scores.append((tst.gunning_fog(text) - MEAN_GRADE) / MEAN_GRADE)
    return scores