import argparse
import sys
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description = 'add description later')

    parser.add_argument(
        '-d',
        '--data_dir',
        type = str, default= 'data/hansards',
        help = 'Data filename prefix (default=data)'
    )

    parser.add_argument(
        '-i',
        '--num_iterations',
        type = int, default= 10,
        help = 'Number of iterations (default=10)'
    )

    parser.add_argument(
        '-e',
        '--english',
        type= str, default = 'e',
        help = 'Suffix of English filename (default=e)'
    )

    parser.add_argument(
        '-f',
        '--french',
        type= str, default='f',
        help= 'Suffix of French filename (default=f)'
    )

    parser.add_argument(
        '-n',
        '--num-sentences',
        type= int, required= True,
        help='Number of sentences to use for training and alignment'
    )

    return parser.parse_args()

class aligner:
    theta = defaultdict(float)
    s_total = defaultdict(int)

    def initProb(self, init_val, bitext, english_words, french_words):
        # for fw in french_words:
        #     for ew in english_words:
        #         tup = (fw, ew) 
        #         self.theta[tup] = init_val

        # Dice initiation
        f_count = defaultdict(int)
        e_count = defaultdict(int)
        fe_count = defaultdict(int)
        for (n, (f, e)) in enumerate(bitext):
            for f_i in set(f):
                f_count[f_i] += 1
                for e_j in set(e):
                    fe_count[(f_i,e_j)] += 1
            for e_j in set(e):
                e_count[e_j] += 1

        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            self.theta[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])

    def condProb(self, num_iterations, bitext, english_words, french_words, english_vocab_size, french_vocab_size):
        for iter in range(num_iterations):          
            # Initialize
            count = {}
            total = {}
            for ew in english_words:
                total[ew] = 0.0
                for fw in french_words:
                    count[(fw, ew)] = 0.0

            for sp in bitext:
                # Normalization
                for fw in sp[0]:
                    self.s_total[fw] = 0.0
                    for ew in sp[1]:
                        self.s_total[fw] += self.theta[(fw, ew)]               
                # Collect counts
                    for ew in sp[1]:
                        count[(fw, ew)] += self.theta[(fw, ew)] / self.s_total[fw]
                        total[ew] += self.theta[(fw, ew)] / self.s_total[fw]

            # Estimate probabilities + smoothing counts
            for ew in english_words:
                for fw in french_words:
                    n = 0.01
                    m = 1
                    if fw == 'NULL':
                        m=5
                    
                    # With smoothing counts
                    self.theta[(fw, ew)] = ((count[(fw, ew)] + n) /(total[ew] + english_vocab_size*n))*m 

                    # Base model
                    # self.theta[(fw, ew)] = ((count[(fw, ew)]) /(total[ew]))*m


    def alignWords(self, bitext):
        for sb in bitext:
            for i, fw in enumerate(sb[0]):
                if fw != 'NULL':
                    best_prob =0
                    best_j= 0
                    for j, ew in enumerate(sb[1]):
                        alpha= 0.8
                        # With distance penalization
                        if self.theta[(fw, ew)]*alpha**(abs(j - i)) > best_prob:
                            best_prob = self.theta[(fw, ew)]*alpha**(abs(j -i))
                            best_j = j

                        # Base model
                        # if self.theta[(fw, ew)] > best_prob:
                        #      best_prob = self.theta[(fw, ew)]
                        #      best_j = j

                    sys.stdout.write("%i-%i " % (i,best_j))
                else:
                    continue
            sys.stdout.write("\n")


    def __init__(self, data_dir, english, french, num_sentences, num_iterations):
        f_data = "%s.%s" % (data_dir, french)
        e_data = "%s.%s" % (data_dir, english)
        bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:num_sentences]

        french_words = []
        english_words = []

        for sp in bitext:
            for ew in sp[1]: 
                english_words.append(ew)
            for fw in sp[0]: 
                french_words.append(fw)

        # Adding NULL words to the source sentence

        # for (n, (f, e)) in enumerate(bitext):
        #     for iter in range(5):
        #         bitext[n][0].append('NULL')

        # french_words.append('NULL')
                
        english_words = sorted(list(set(english_words)), key=lambda s: s.lower()) 
        french_words = sorted(list(set(french_words)), key=lambda s: s.lower())

        english_vocab_size = len(english_words)
        french_vocab_size = len(french_words)

        init_val = 1.0 / french_vocab_size

        self.initProb(init_val, bitext, english_words, french_words)

        self.condProb(num_iterations, bitext, english_words, french_words, english_vocab_size, french_vocab_size)

        self.alignWords(bitext)
   
def main():
    
    args = parse_args() 

    aligner(args.data_dir, args.english, args.french, args.num_sentences,args.num_iterations)



if __name__ == "__main__":
    main()