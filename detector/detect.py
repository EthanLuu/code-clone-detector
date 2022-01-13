import torch
from encode import ast_to_blocks
from pre_process import code_to_ast
from settings import settings
import dill
from gensim.models.word2vec import Word2Vec


loss_function = torch.nn.BCELoss()
categories = 5


class Detector:
    def __init__(self):
        self.models_dic = self.get_models_dic()
        word2vec = Word2Vec.load(settings.w2v_model_path).wv
        self.vocab = word2vec.vocab
        self.max_token = word2vec.syn0.shape[0]
        pass

    def get_models_dic(self):
        models_dic = dict()
        for t in range(1, categories + 1):
            model_path = settings.models_path + "/model_" + str(t) + ".pkl"
            f = open(model_path, 'rb')
            model = dill.load(f)
            f.close()
            models_dic[t] = model
        return models_dic

    def detect_by_type(self, codeX, codeY, t):
        print("Testing-%d..." % t)
        model_path = settings.models_path + "/model_" + str(t) + ".pkl"
        f = open(model_path, 'rb')
        model = dill.load(f)
        f.close()
        model.batch_size = 1
        model.hidden = model.init_hidden()
        astX = code_to_ast(codeX)
        astY = code_to_ast(codeY)

        blocksX = ast_to_blocks(astX, self.vocab, self.max_token)
        blocksY = ast_to_blocks(astY, self.vocab, self.max_token)
        output = model([blocksX], [blocksY])
        print(output)
        return output.data

    def detect(self, codeX, codeY):
        res = []
        for t in range(1, categories + 1):
            data = self.detect_by_type(codeX, codeY, t)
            res.append(data.item() > 0.5)
        print(res)
        return res


def main():
    codeX = '''
    class Solution {
        public int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> map = new HashMap<>();
            for(int i = 0; i< nums.length; i++) {
                if(map.containsKey(target - nums[i])) {
                    return new int[] {map.get(target-nums[i]),i};
                }
                map.put(nums[i], i);
            }
            throw new IllegalArgumentException("No two sum solution");
        }
    }
    '''
    codeY = '''
    class Solution {
        public int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
            for (int i = 0; i < nums.length; ++i) {
                if (hashtable.containsKey(target - nums[i])) {
                    return new int[]{hashtable.get(target - nums[i]), i};
                }
                hashtable.put(nums[i], i);
            }
            return new int[0];
        }
    }
    '''
    detector = Detector()
    detector.detect(codeX, codeY)


if __name__ == "__main__":
    main()
