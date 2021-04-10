from evaluation.importance_weights import highest_sim_retrieval, cft_weights
from evaluation.visualization import text_highlighter

def cft_retrieval_printer(query, representations, model, vocab, top_k=5, max_alpha=0.8, temp=1.0):

    neighbours = highest_sim_retrieval(query=query,
                                       representations=representations,
                                       model=model, vocab=vocab,
                                       top_k=top_k)

    print("Query:")
    text_highlighter(text=query,
                     weights=cft_weights(query, model, vocab),
                     verbose=True,
                     max_alpha=max_alpha,
                     temp=temp)
    print("\nMost similar SNLI sentences:")
    for neighnour in neighbours:
        text_highlighter(text=neighnour,
                         weights=cft_weights(neighnour, model, vocab),
                         verbose=True,
                         max_alpha=max_alpha,
                         temp=temp)
        print("")
