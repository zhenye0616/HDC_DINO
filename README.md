use this command to create the env
-conda env create -f environment.yml

After that activate the env and compile the CUDA operators
-cd models/dino/ops
-python setup.py build install
-# unit test (should see all checking is True)
-python test.py
-cd ../../..


Pretrained checkpoint is in /checkpoint
Fintuning code with hdc head is in dino_hd.py

whenever you want to change the classification head or load checkpoint make sure to go to dino.py or else there will be mismatch between checkpoint weights and the model.

""      _class_embed = nn.Linear(hidden_dim, num_classes) (linear)
        #_class_embed = MLP(hidden_dim, hidden_dim, num_classes, num_layers=3) (mlp)
        #_class_embed = HD(num_classes, hidden_dim, dim = 100).to('cuda') (HDC)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01  (linear)
        bias_value = -math.log((1 - prior_prob) / prior_prob) (linear)
        #HD_init(_class_embed) (HDC)
        #nn.init.constant_(_class_embed.layers[-1].bias, bias_value) #init mlp classifier (mlp)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value (linear)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)""

run dino_video.ipynb to generate visual results
