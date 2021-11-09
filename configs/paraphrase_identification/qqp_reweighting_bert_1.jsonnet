local transformer_model = "bert-base-uncased";
local transformer_dim = 768;

{
  "dataset_reader": {
    "type": "weighted_qqp",
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
  "train_data_path": "data/paraphrase_identification/weighted_qqp.train.jsonl",
  "validation_data_path": "data/paraphrase_identification/qqp.dev.jsonl",
  "test_data_path": "data/paraphrase_identification/paws.dev_and_test.jsonl",
  "model": {
    "type": "sample_weight_basic_classifier",
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
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
    },
    "feedforward": {
      "input_dim": transformer_dim,
      "num_layers": 1,
      "hidden_dims": transformer_dim,
      "activations": "tanh"
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32 
    }
  },
  "trainer": {
    "num_epochs": 4,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 5e-5,
      "weight_decay": 0.1,
    },
    "use_amp": true,
    "cuda_device" : 0,
  }
}