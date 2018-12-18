import os
from chainer import serializers
from net import Transformer
import preprocess

model = Transformer(2, 38, 38,
        256,
        h=4,
        dropout=0.1,
        max_length=500,
        use_label_smoothing=False,
        embed_position=False)
serializers.load_npz('../../result/Transformer/model_iter_706000.npz', model)

en_path = os.path.join('../../data/Enum2Enum', 'sval.txt')
source_vocab = ['<eos>', '<unk>', '<bos>'] + \
    preprocess.count_words(en_path, 50)
source_ids = {word: index for index, word in enumerate(source_vocab)}
source_words = {i: w for w, i in source_ids.items()}

source = 'c '*120
words = preprocess.split_sentence(source)
print('# source : ' + ' '.join(words))
x = model.xp.array([source_ids.get(w, 1) for w in words], 'i')
h = model.encode([x])
print(h)
#model.translate([x], beam=5)