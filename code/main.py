import argparse
import torch.optim as optim
import torch
import random
import numpy as np
import os
from source.model import Seq2Seq
from utils.vocab import Vocab
from utils.dataloader import BBCLoader, DuconvLoader, SampleGenerator, build_feed_data
from utils.loss import compute_loss
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import json

use_gpu = torch.cuda.is_available()

def str2bool(v):
    """ str2bool """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)


def topic_recovery(string, topic_dict):
    for key in topic_dict:
        if key in string:
            string =string.replace(key, topic_dict[key])
    return string

def model_summary(model):
    print('Model Structure:')
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params:{total_params}")


def Config():
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group('Data')
    # data path config
    data_arg.add_argument('--data_class', type=str, default='duconv')
    data_arg.add_argument('--train_data_path', type=str, default='../data/duconv/demo.train')
    data_arg.add_argument('--dev_data_path', type=str, default='../data/duconv/demo.dev')
    data_arg.add_argument('--test_data_path', type=str, default='../data/duconv/demo.test')
    data_arg.add_argument('--test_sample_path', type=str, default='../data/duconv/sample.test.json')
    data_arg.add_argument('--test_topic_path', type=str, default='../data/duconv/demo.test.topic')

    data_arg.add_argument('--vocab_path', type=str, default='../data/duconv/vocab.txt')
    data_arg.add_argument('--vocab_size', type=int, default=50000)

    # file path to save model checkpoint during training & generation result during testing
    data_arg.add_argument('--checkpoint_dir', type=str, default='./checkpoints/duconv')
    data_arg.add_argument('--output_dir', type=str, default='./outputs/duconv')

    # Network
    net_arg = parser.add_argument_group('Network')
    net_arg.add_argument('--share_embedding', type=str2bool, default=True)  # encoder and decoder share the embedding layer or not
    net_arg.add_argument('--embed_size', type=str, default=200)
    net_arg.add_argument('--hidden_size', type=str, default=400)
    net_arg.add_argument('--dropout', type=float, default=0.3)

    # Training
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--run_type', type=str, default='train')
    train_arg.add_argument('--stage', type=int, default=0)
    # init model/optimizer path, in order to restore training process from specific checkpoints
    train_arg.add_argument('--init_model_path', type=str, default=None)
    # training config
    train_arg.add_argument('--lr', type=float, default=0.0005)
    train_arg.add_argument('--max_grad_norm', type=float, default=5.0)
    train_arg.add_argument('--num_epoch', type=int, default=12)
    train_arg.add_argument('--batch_size', type=int, default=32)

    # Generation
    gen_arg = parser.add_argument_group('Generation')
    gen_arg.add_argument("--decode_mode", type=str, default='greedy')  # use beam search or not, value = 'greedy' or 'beam'
    gen_arg.add_argument("--beam_size", type=int, default=5)
    gen_arg.add_argument("--max_dec_len", type=int, default=100)

    # Other
    misc_arg = parser.add_argument_group('Misc')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--log_steps', type=int, default=100)
    misc_arg.add_argument("--valid_steps", type=int, default=800)

    config = parser.parse_args()

    return config


class Runner():
    def __init__(self, config):
        print(config)
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.config = config
        self.config.unk_id = self.vocab.unk_id
        if config.stage == 0:
            self.pre_train = True
        else:
            self.pre_train = False

    def setup_train(self):
        # Data loader
        if config.data_class == 'wizard':
            self.train_loader = BBCLoader(
                vocab=self.vocab,
                data_path=config.train_data_path,
                batch_size=config.batch_size,
                shuffle=True
            )
            self.dev_loader = BBCLoader(
                vocab=self.vocab,
                data_path=config.dev_data_path,
                batch_size=config.batch_size,
                shuffle=False
            )
        elif config.data_class == 'duconv':
            self.train_loader = DuconvLoader(
                vocab=self.vocab,
                data_path=config.train_data_path,
                batch_size=config.batch_size,
                shuffle=True
            )
            self.dev_loader = DuconvLoader(
                vocab=self.vocab,
                data_path=config.dev_data_path,
                batch_size=config.batch_size,
                shuffle=False
            )
        # checkpoint dir
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # init model
        if use_gpu:
            self.model = Seq2Seq(config).cuda()
        else:
            self.model = Seq2Seq(config)
        model_summary(self.model)
        
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=config.lr)
        self.start_epoch, self.start_step = 0, 0
        if config.init_model_path is not None:
            print('load model from %s' % config.init_model_path)
            state = torch.load(config.init_model_path, map_location=lambda storage, location: storage)
            self.start_epoch = state['epoch'] + 1
            self.start_step = state['step']
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        self.model.train()

    def setup_test(self):
        # Data loader
        if config.data_class == 'wizard':
            self.test_loader = BBCLoader(
                vocab=self.vocab,
                data_path=config.test_data_path,
                batch_size=config.batch_size,
                shuffle=False
            )
            self.test_sample_loader = SampleGenerator(
                data_path=self.config.test_sample_path,
                batch_size=self.config.batch_size,
            )
            self.test_topic_loader = SampleGenerator(
                data_path=None,
                batch_size=self.config.batch_size,
                block_data=self.test_sample_loader.data_size * [0]
            )
        elif config.data_class == 'duconv':
            self.test_loader = DuconvLoader(
                vocab=self.vocab,
                data_path=self.config.test_data_path,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            self.test_sample_loader = SampleGenerator(
                data_path=self.config.test_sample_path,
                batch_size=self.config.batch_size
            )
            self.test_topic_loader = SampleGenerator(
                data_path=self.config.test_topic_path,
                batch_size=self.config.batch_size
            )
        # output dir
        self.output_dir = config.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # init model
        if use_gpu:
            self.model = Seq2Seq(config).cuda()
        else:
            self.model = Seq2Seq(config)
        if config.init_model_path is not None:
            print('load model from %s' % config.init_model_path)
            state = torch.load(config.init_model_path, map_location=lambda storage, location: storage)
            for item in state['model']:
                print(item)
            return
            self.model.load_state_dict(state['model'])
            self.model.eval()

    def save_model(self, epoch, step, eval_loss, is_best=False, end_pretrain=False):
        state = {
            'epoch': epoch,
            'step': step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': eval_loss
        }
        if not end_pretrain:
            model_save_path = os.path.join(config.checkpoint_dir, 'epoch_%d_iter_%d_nll_%.5f'%(epoch, step, eval_loss))
            torch.save(state, model_save_path)
        else:
            model_save_path = os.path.join(config.checkpoint_dir, 'model_stage_0')
            torch.save(state, model_save_path)
            return
        if is_best:
            model_save_path = os.path.join(config.checkpoint_dir, 'best_model')
            torch.save(state, model_save_path)

    def run_one_batch(self, feed_data, is_train=True):
        self.optimizer.zero_grad()
        
        bow_probs, decode_probs, prior_dist, posterior_dist = \
            self.model(src_input=feed_data['src_input'], src_len=feed_data['src_len'], src_padding_mask=feed_data['src_padding_mask'],
                    kg_input=feed_data['kg_input'], kg_len=feed_data['kg_len'], kg_mask=feed_data['kg_mask'],
                    kg_padding_mask=feed_data['kg_padding_mask'], kg_extend_vocab=feed_data['kg_extend_vocab'], extra_zero=feed_data['extra_zero'],
                    tgt_input=feed_data['tgt_input'], tgt_len=feed_data['tgt_len'], is_train=True)

        bow_loss, kl_loss, nllloss, final_loss = compute_loss(bow_probs=bow_probs, bow_label=feed_data['tgt_input'][:,1:],
                 decode_probs=decode_probs, tgt_label=feed_data['label'],
                 prior_attn=prior_dist, posterior_att=posterior_dist, kl_and_nll_factor=feed_data['kl_and_nll_factor'])
        if is_train:
            if self.pre_train:
                bow_loss.backward()
                clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()
            else:
                final_loss.backward()
                clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()

        return nllloss.item(), bow_loss.item(), kl_loss.item(), final_loss.item()

    def trainIter(self):
        total_step = self.start_step
        log_steps = config.log_steps
        valid_steps = config.valid_steps
        best_score = 10000000
        batch_num = len(self.train_loader())
        start = time.time()
        for epoch_idx in range(self.start_epoch, config.num_epoch):
            total_nll_loss = 0
            total_bow_loss = 0
            total_kl_loss = 0
            total_final_loss = 0
            sample_num = 0
            for batch_idx, batch_data in enumerate(self.train_loader()):
                feed_data = build_feed_data(batch_data, use_gpu, self.pre_train)
                nllloss, bow_loss, kl_loss, final_loss = self.run_one_batch(feed_data, is_train=True)
                total_step += 1
                sample_num += 1
                total_nll_loss += nllloss
                total_bow_loss += bow_loss
                total_kl_loss += kl_loss
                total_final_loss += final_loss
                if (batch_idx + 1) % log_steps == 0:
                    print("Train epoch %d step %d/%d | nll loss %0.4f, bow loss %0.4f, kl loss %0.4f, final loss %0.4f, time consuming %d"
                          % (epoch_idx, batch_idx + 1, batch_num, total_nll_loss / sample_num, total_bow_loss / sample_num,
                          total_kl_loss / sample_num, total_final_loss / sample_num, time.time() - start))
                    total_nll_loss = 0
                    total_bow_loss = 0
                    total_kl_loss = 0
                    total_final_loss = 0
                    sample_num = 0
                
                if total_step > 0 and total_step % valid_steps == 0:
                    if not self.pre_train:
                        nllloss, bow_loss, kl_loss, final_loss  = self.devIter()
                        if nllloss < best_score:
                            best_score = nllloss
                            self.save_model(
                                epoch_idx, total_step, nllloss, is_best=True
                            )
            # After one epoch    
            if not self.pre_train:
                nllloss, bow_loss, kl_loss, final_loss = self.devIter()
                if nllloss < best_score:
                    best_score = nllloss
                    self.save_model(
                        epoch_idx, total_step, nllloss, is_best=True
                    )
                else:
                    self.save_model(
                        epoch_idx, total_step, nllloss, is_best=False
                    )
        if self.pre_train:
            self.save_model(
                        epoch_idx, total_step, nllloss, is_best=False, end_pretrain=True
                    )


    def devIter(self):
        self.model.eval()
        
        total_nll_loss = 0
        total_bow_loss = 0
        total_kl_loss = 0
        total_final_loss = 0

        sample_num = 0
        with torch.no_grad():
            for batch_data in self.dev_loader():
                feed_data = build_feed_data(batch_data, use_gpu=use_gpu, pre_train=self.pre_train)
                nllloss, bow_loss, kl_loss, final_loss = self.run_one_batch(feed_data, False)
                
                real_batch_size = feed_data['src_len'].size(0)
                sample_num += real_batch_size
                total_nll_loss += nllloss * real_batch_size
                total_bow_loss += bow_loss * real_batch_size
                total_kl_loss += kl_loss * real_batch_size
                total_final_loss += final_loss * real_batch_size

        print("Eval | nll loss %0.4f, bow loss %0.4f, kl loss %0.4f, final loss %0.4f"
                          % (total_nll_loss / sample_num, total_bow_loss / sample_num,
                          total_kl_loss / sample_num, total_final_loss / sample_num))
        self.model.train()
        return total_nll_loss / sample_num, total_bow_loss / sample_num, total_kl_loss / sample_num, total_final_loss / sample_num


    def testIter(self):
        eos_id = self.vocab.word2id('<eos>')
        fw_eval = open(os.path.join(self.output_dir, 'eval.txt'), 'w', encoding='utf8')
        fw_example = open(os.path.join(self.output_dir, 'example.txt'), 'w', encoding='utf8')
        examples = []
        with torch.no_grad():
            for batch_data, sample_data, topic_data in zip(self.test_loader(), self.test_sample_loader(), self.test_topic_loader()):
                oov_words = batch_data[-1]
                feed_data = build_feed_data(batch_data, use_gpu=use_gpu, pre_train=self.pre_train)
                pred_ids, ks_dist = self.model(
                    src_input=feed_data['src_input'], src_len=feed_data['src_len'], src_padding_mask=feed_data['src_padding_mask'],
                    kg_input=feed_data['kg_input'],kg_len=feed_data['kg_len'], kg_mask=feed_data['kg_mask'],
                    kg_padding_mask=feed_data['kg_padding_mask'], kg_extend_vocab=feed_data['kg_extend_vocab'], extra_zero=feed_data['extra_zero'],
                    tgt_input=feed_data['tgt_input'], tgt_len=feed_data['tgt_len'], is_train=False
                )
               
                pred_words = []
                for i, pred_sample in enumerate(pred_ids.tolist()):
                    if eos_id in pred_sample:
                        eos_index = pred_sample.index(eos_id)
                        pred_words.append(self.vocab.outputids2words(pred_sample[:eos_index], oov_words[i]))
                    else:
                        pred_words.append(self.vocab.outputids2words(pred_sample, oov_words[i]))
                select_kg_prob, select_kg_index = ks_dist.topk(k=3, dim=1)
                select_kg_prob = select_kg_prob.tolist()
                select_kg_index = select_kg_index.tolist()
                for i, origin_sample in enumerate(sample_data):
                    kg = origin_sample['knowledge']
                    real_kg_num = len(kg)
                    selected_kg = []
                    for prob, index in zip(select_kg_prob[i], select_kg_index[i]):
                        if config.data_class == 'duconv':
                            if index < real_kg_num:
                                selected_kg.append(
                                    {
                                        'name': kg[index][0],
                                        'attrname': kg[index][1],
                                        'attrvalue': kg[index][2],
                                        'prob': prob
                                    }
                                )
                            else:
                                selected_kg.append(
                                    {
                                        'name': '<pad>',
                                        'attrname': '<pad>',
                                        'attrvalue': '<pad>',
                                        'prob': prob
                                    }
                                )
                        else:
                            if index < real_kg_num:
                                selected_kg.append(
                                    {
                                        'knowledge': kg[index],
                                        'prob': prob
                                    }
                                )
                            else:
                                selected_kg.append(
                                    {
                                        'knowledge': '<pad>',
                                        'prob': prob
                                    }
                                )
                    if config.data_class == 'duconv':
                        eval_line = topic_recovery(' '.join(pred_words[i]), topic_data[i]).lower() + '\t' + origin_sample['response'].lower()
                        examples.append(
                            {       
                                'history': origin_sample['history'],
                                'response': topic_recovery(' '.join(pred_words[i]), topic_data[i]),
                                'reference': origin_sample['response'],
                                'selected_attrs': selected_kg,
                            }
                        )
                    else:
                        eval_line = ' '.join(pred_words[i]).lower() + '\t' + origin_sample['response'].lower()
                        examples.append(
                            {       
                                'history': origin_sample['history'],
                                'response': ' '.join(pred_words[i]),
                                'reference': origin_sample['response'],
                                'selected_attrs': selected_kg,
                            }
                        )
                    fw_eval.write(eval_line + '\n')
        fw_eval.close()
        json.dump(examples, fw_example, ensure_ascii=False, indent=3)
        fw_example.close()

if __name__ == '__main__':
    config = Config()
    runner = Runner(config)
    run_type = config.run_type
    if run_type == "train":
        runner.setup_train()
        runner.trainIter()
    elif run_type == 'test':
        runner.setup_test()
        runner.testIter()