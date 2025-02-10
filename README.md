# Prompts

### Zero-Shot

You are a highly trained text data annotation tool capable of
    providing subjective responses. Rate the semantic similarity of the target word in these sentences 1 and 2. Consider
    only the objects/concepts the word forms refer to: ignore any common etymology and
    metaphorical similarity! Ignore case! Ignore number (cat/Cats = identical meaning). Homonyms (like bat the animal vs
    bat in baseball) count as unrelated. Output numeric rating: 1 is unrelated; 2 is distantly
    related; 3 is closely related; 4 is identical meaning.Your response should align with a
    human’s succinct judgment.

    Given the following sentence pair score according the explained, give
    only a numeric value as answer and do not explain why the chosen score:
    
    first sentence: {sentence1}
    second sentence: {sentence2}
    target word: {target_word}
    score:

  ### Few-Shot

  You are a highly trained text data annotation tool capable of
    providing subjective responses. Rate the semantic similarity of the target word in these sentences 1 and 2. Consider
    only the objects/concepts the word forms refer to: ignore any common etymology and
    metaphorical similarity! Ignore case! Ignore number (cat/Cats = identical meaning). Homonyms (like bat the animal vs
    bat in baseball) count as unrelated. Output numeric rating: 1 is unrelated; 2 is distantly
    related; 3 is closely related; 4 is identical meaning.Your response should align with a
    human’s succinct judgment.
    
    Examples of sentences for each possible annotation:

    Example:
        target word: bank
        first sentence: His parents had left a lot of money in the bank and now it was all Measle's, but a judge had said that 
            Measle was too young to get it.
        second sentence: Sherrell, it is said, was sitting on the bank of the river close by, and as soon as the men had disappeared
            from sight he jumped on board the schooner.
        Score: 1

    Example:
        target word: crossroad
        first sentence: He came to a crossroad and read the signs; to the south, 
            Kenniston, 20 m.
        second sentence: As a result we are at a crossroad; either school integration efforts will
            be abandoned in the South, or they will be pursued in the Nort as well.
        Score: 2

    Example:
        target word: child
        first sentence: He agreed and began practicing his sleight of hand tricks to the great pleasure of some children,
            the same ones, I suspect, who had plagued me when I was a child.
        second sentence: The daylight had long faded; her child lay camly sleeping by her side; a candle was burning dimly
            on the stand.
        Score: 3

    Example:
        target word: eat
        first sentence: Speaking of bread and butter reminds me that we'd better eat ours before the coffee get quite cold.
        second sentence: When the meal was over and they had finished their tea after they ate, Wang the Second took the trusty
            man to his elder brother's gate.
        Score: 4
            
        
    Given the following sentence pair score according the explained, give
    only a numeric value as answer and do not explain why the chosen score:
    
    target word: {target_word}
    first sentence: {sentence1}
    second sentence: {sentence2}
    score:

### Chain of Thought

 You are a highly trained text data annotation tool capable of
    providing subjective responses. Rate the semantic similarity of the target word in these sentences 1 and 2. Consider
    only the objects/concepts the word forms refer to: ignore any common etymology and
    metaphorical similarity! Ignore case! Ignore number (cat/Cats = identical meaning). Homonyms (like bat the animal vs
    bat in baseball) count as unrelated. Output numeric rating: 1 is unrelated; 2 is distantly
    related; 3 is closely related; 4 is identical meaning.Your response should align with a
    human’s succinct judgment.
    
    Example:
        target word: bank
        first sentence: His parents had left a lot of money in the bank and now it was all Measle's, but a judge had said that 
            Measle was too young to get it.
        second sentence: Sherrell, it is said, was sitting on the bank of the river close by, and as soon as the men had disappeared
            from sight he jumped on board the schooner.
        
        Score: 1, let's think step by step, the annotation is 1 because in the first sentence "bank" is related to
            financial institutions, where people can deposit or withdraw their money, while in the
            second sentence "bank" refers to a bench or seat, typically found in parks or public spaces, where people can sit and rest.
    
    Example:
        target word: tree
        first sentence: A binary tree is a data structure for storing organized data.
            second sentence: The trees in that forest are rich in aroma and colors.
        
        Score: 2, let's think step by step, The annotation is 2 because in the first
            sentence "tree" means a data structure in computing, and in the second
            sentence "trees" means the plant that has leaves. The meaning is loosely
            related because they share the shape of a tree, but in one case it is a data structure and in the second sentence, it is a tree.
            
    Example:
        target word:
        first sentence: With this well-known program, we will be able to manipulate the data through its cells.
        second sentence: As a result of this analysis, we will obtain a different map, a map where each visible
            cell comes with a number of 0 or 180 degrees.
        
        Score: 3, let's think step by step, the annotation is 3 because in the first sentence the meaning of
            "cell" is related to grids where you can input information, and in the second sentence the meaning
            is also related to grids where you can gather information. It is not 4 because the meanings differ
            in that in one sentence they are used in a program and in the other sentence they are used in a map.
    
    Example:
        target word: horse
        first sentence: Marti's white horse gallops to victory.
        second sentence: When horses are in a herd, they protect each other.

        Score: 4, let's think step by step, in this case, the annotation is 4 because in both sentences, the meaning of
            horse y horses is related to the animal. Please note that horse is singular and horses
            is plural do not affect the annotation.
    
    Given the following sentence pair score according the explained, give
    only a numeric value as answer and do not explain why the chosen score:

    target word: {target_word}
    first sentence: {sentence1}
    second sentence: {sentence2}
    score:
    

