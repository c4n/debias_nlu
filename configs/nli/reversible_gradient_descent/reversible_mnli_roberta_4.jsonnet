local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader": {
    "type": "reversible_snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": "spurious_corr/MNLI/data/multinli_1.0_train.jsonl",
  // "train_data_path": "spurious_corr/MNLI/data/multinli_small_train.jsonl",
  "validation_data_path": "spurious_corr/MNLI/data/multinli_1.0_dev_matched.jsonl",
  "test_data_path": "spurious_corr/MNLI/data/multinli_1.0_dev_mismatched.jsonl",
  "model": {
    "type": "adv_basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1,
    "namespace": "tags",
    "_lambda" : 4.0
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 16 
    }
  },
  
  "trainer": {
   "type": "reversible_gradient_descent",
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06,
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-6,
      "weight_decay": 0.1,
    },
    "cuda_device" : 0
  }
}
