class settings:
    data_path = "./data"
    
    train_path = "./data/train"
    test_path = "./data/test"
    
    train_pairs_path = train_path + "/java_pairs.pkl"
    vec_size = 128
    
    java_source_path = "./assets/java_source.tsv"
    java_pairs_path = "./assets/java_pairs.pkl"
    java_ast_path = "./data/java_ast.pkl"
    java_block_path = "./data/java_blocks.pkl"
    
    train_block_path = train_path + "/java_blocks.pkl"
    test_block_path = test_path + "/java_blocks.pkl"
    test_pairs_path = test_path + "/java_pairs.pkl"
    
    models_path = "./models"
    w2v_model_path = models_path + "/w2v_" + str(vec_size)