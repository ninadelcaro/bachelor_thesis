import pandas as pd


# MERGE MECO DF WITH RNN RESULTS

en_merged = pd.read_csv("en_merged.csv")
rnn_output = pd.read_csv("final_models\\model_en_0delay_seed_2342_epoch_41\\accuracies_test_on_meco_v2.csv")
drnn_output = pd.read_csv("final_models\\model_en_1delay_seed_2342_epoch_45\\accuracies_test_on_meco_v2.csv")

# remove the end of sentence tag for the target word
rnn_output = rnn_output[rnn_output['actual_word'] != '</s>'].reset_index()
rnn_output.rename(columns={"correct": "rnn_correct", "previous_word": "rnn_previous_word",
                           "predicted_word": "rnn_predicted_word", "entropy": "rnn_entropy",
                           "entropy_top10": "rnn_entropy_top10", "surprisal": "rnn_surprisal",
                           "target_in_top10": "rnn_target_in_top10", "perplexity_per_sentence": "rnn_perplexity"}, inplace=True)
rnn_output_merged_with_lemmas = pd.concat([en_merged, rnn_output], axis=1)

drnn_output = drnn_output[drnn_output['actual_word'] != '</s>'].reset_index()
drnn_output.rename(columns={"correct": "drnn_correct", "previous_word": "drnn_previous_word",
                           "predicted_word": "drnn_predicted_word", "entropy": "drnn_entropy",
                           "entropy_top10": "drnn_entropy_top10", "surprisal": "drnn_surprisal",
                           "target_in_top10": "drnn_target_in_top10", "perplexity_per_sentence": "drnn_perplexity"}, inplace=True)
rnn_results_and_lemmas = pd.concat([rnn_output_merged_with_lemmas, drnn_output[["drnn_correct",
                                                                                "drnn_previous_word",
                                                                                "drnn_predicted_word",
                                                                                "drnn_entropy",
                                                                                "drnn_entropy_top10",
                                                                                "drnn_surprisal",
                                                                                "drnn_target_in_top10",
                                                                                "drnn_perplexity"]]], axis=1)


df_en_meco = pd.read_csv('df_en_meco.csv')

test = df_en_meco.merge(rnn_results_and_lemmas, how='left', on=['text_id', 'total_word_idx'])
# set of participants for which merge doesnt work
part_list = pd.unique(test[test['word_x'] != test['word_y']]['participant']) 
df_en_meco_1 = df_en_meco[df_en_meco['participant'].isin(part_list)]
df_en_meco_2 = df_en_meco[~df_en_meco['participant'].isin(part_list)]

rnn_results_and_lemmas_2 = rnn_results_and_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)
rnn_results_and_lemmas_1 = rnn_results_and_lemmas.drop(['sent_id_and_idx', 'word_idx'], axis=1)
# change the rnn df for the first set of participants
rnn_results_and_lemmas_1_messed_up_part = rnn_results_and_lemmas_1[(rnn_results_and_lemmas_1['text_id'] == 3) 
                                                                   & (rnn_results_and_lemmas_1['total_word_idx'] > 148)].copy()
rnn_results_and_lemmas_1_not_messed_up_part = rnn_results_and_lemmas_1[((rnn_results_and_lemmas_1['text_id'] != 3) 
                                                                   | (rnn_results_and_lemmas_1['total_word_idx'] <= 148))].copy()
rnn_results_and_lemmas_1_messed_up_part['total_word_idx'] = rnn_results_and_lemmas_1_messed_up_part['total_word_idx'] - 1
rnn_results_and_lemmas_1 = pd.concat([rnn_results_and_lemmas_1_messed_up_part, rnn_results_and_lemmas_1_not_messed_up_part])

test1 = df_en_meco_1.merge(rnn_results_and_lemmas_1, how='left', on=['text_id', 'total_word_idx'])
test2 = df_en_meco_2.merge(rnn_results_and_lemmas_2, how='left', on=['text_id', 'total_word_idx'])

df_meco_rnn = pd.concat([test1, test2])
# this needs to be 0
print(sum(df_meco_rnn['word_x'] != df_meco_rnn['word_y']))
df_meco_rnn['word'] = df_meco_rnn['word_x']
df_meco_rnn.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_meco_rnn.to_csv('df_meco_rnn.csv')


# MERGE POTSDAM DF WITH RNN RESULTS

hi_merged = pd.read_csv("hi_merged.csv")
rnn_output = pd.read_csv("final_models\\model_hi_0delay_seed_2342_epoch_45\\accuracies_test_on_potsdam_v2.csv")
drnn_output = pd.read_csv("final_models\\model_hi_1delay_seed_2342_epoch_45\\accuracies_test_on_potsdam_v2.csv")

# remove the end of sentence tag for the target word
rnn_output = rnn_output[rnn_output['actual_word'] != '</s>'].reset_index()
rnn_output.rename(columns={"correct": "rnn_correct", "previous_word": "rnn_previous_word",
                           "predicted_word": "rnn_predicted_word", "entropy": "rnn_entropy",
                           "entropy_top10": "rnn_entropy_top10", "surprisal": "rnn_surprisal",
                           "target_in_top10": "rnn_target_in_top10", "perplexity_per_sentence": "rnn_perplexity"}, inplace=True)
rnn_output_merged_with_lemmas = pd.concat([hi_merged, rnn_output], axis=1)

drnn_output = drnn_output[drnn_output['actual_word'] != '</s>'].reset_index()
drnn_output.rename(columns={"correct": "drnn_correct", "previous_word": "drnn_previous_word",
                           "predicted_word": "drnn_predicted_word", "entropy": "drnn_entropy",
                           "entropy_top10": "drnn_entropy_top10", "surprisal": "drnn_surprisal",
                           "target_in_top10": "drnn_target_in_top10", "perplexity_per_sentence": "drnn_perplexity"}, inplace=True)
rnn_results_and_lemmas = pd.concat([rnn_output_merged_with_lemmas, drnn_output[["drnn_correct",
                                                                                "drnn_previous_word",
                                                                                "drnn_predicted_word",
                                                                                "drnn_entropy",
                                                                                "drnn_entropy_top10",
                                                                                "drnn_surprisal",
                                                                                "drnn_target_in_top10",
                                                                                "drnn_perplexity"]]], axis=1)


df_hi_potsdam = pd.read_csv('df_hi_potsdam.csv')
df_potsdam_rnn = df_hi_potsdam.merge(rnn_results_and_lemmas, how='left', on=['sent_id_and_idx', 'word_idx'])
df_potsdam_rnn.drop(["entropy", "surprisal"], axis=1, inplace=True)
# this should be 0
print(sum(df_potsdam_rnn['word_x'] != df_potsdam_rnn['word_y']))
df_potsdam_rnn['word'] = df_potsdam_rnn['word_x']
df_potsdam_rnn.drop(['word_x', 'word_y'], axis=1, inplace=True)
df_potsdam_rnn.to_csv('df_potsdam_rnn.csv')


# MERGE THE TWO DATAFRAMES INTO ONE SPECIFICALLY FOR R ANALYSIS
print(set(df_potsdam_rnn.columns).difference(set(df_meco_rnn.columns)))
print(set(df_meco_rnn.columns).difference(set(df_potsdam_rnn.columns)))

r_analysis_df =pd.concat([df_potsdam_rnn, df_meco_rnn], join='outer', axis=0, ignore_index=True)
r_analysis_df.to_csv('r_analysis_df.csv')