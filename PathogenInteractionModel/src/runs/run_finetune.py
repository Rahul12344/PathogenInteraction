import json
import os
import tensorflow as tf
import tensorflow_models as tfm


bert_config_file = os.path.join("/Users/rahulnatarajan/EC2_data/OrganismInteraction/pretrained_models/bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

encoder_config = tfm.nlp.encoders.EncoderConfig({
    'type':'bert',
    'bert': config_dict
})
bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)


tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join("/Users/rahulnatarajan/EC2_data/OrganismInteraction/pretrained_models/pretrained_models/vocab.txt"),
    lower_case=True)

bert_classifier = tfm.nlp.models.BertClassifier(network=bert_encoder, num_classes=2)


bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)

class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):
        tok1 = self.tokenizer(inputs['sentence1'])
        tok2 = self.tokenizer(inputs['sentence2'])

        packed = self.packer([tok1, tok2])

        if 'label' in inputs:
            return packed, inputs['label']
        else:
            return packed