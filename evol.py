from __future__ import division
import random
import bisect
import collections
import itertools
import copy
import ast
import argparse
import sys

BASES = "AGCT"
TRANSITIONS = {'A':'G','G':'A','C':'T','T':'C'}
TRANSVERSIONS = {'A':'CT','G':'CT','C':'AG','T':'AG'}


def generate_rand_genome(genLen):
    return [ random.choice(BASES) for _ in range(genLen) ]


def mutate_base(base, tiP):
    if random.random() < tiP:
        return TRANSITIONS[base]
    else:
        return random.choice(TRANSVERSIONS[base])


def parentIndexGenerator(recRate):
    ind = 0
    while True:
        ind = (ind+1)%2 if random.random() < recRate else ind
        yield ind


class Organism(object):
    def __init__(self, genome, sex):
        self.genome = genome
        self.sex = sex
        self.age = 0
        self.weight = 1
        self.youngestChildAge = 1000
        self.nChildren = 0
    
    def set_weight(self, posBaseWeights):
        self.weight = sum(posBaseWeights[b][i]
                for i,b in enumerate(self.genome))

    def mutate(self, mutP, tiP):
        self.genome = [ mutate_base(b,tiP) if random.random() < mutP else b
            for b in self.genome ]

    def recombine(self, org, recRate):
        parInd = parentIndexGenerator(recRate)
        self.genome = [ bb[next(parInd)]
            for bb in zip(self.genome, org.genome) ]

    def get_child(self, sex, mutP, tiP, posBaseWeights,
        recRate=0, secondParentOrg=None):
        self.youngestChildAge = 0
        self.nChildren += 1
        child = Organism(self.genome[:], sex)
        if not secondParentOrg is None:
            child.recombine(secondParentOrg, recRate)
            secondParentOrg.nChildren += 1
            if RECORD:
                child.set_weight(posBaseWeights)
                DATA_FILE_SEXUAL_REC.write("%s\n" % child.weight)
        child.mutate(mutP, tiP)
        child.set_weight(posBaseWeights)
        if RECORD:
            if not secondParentOrg is None:
                DATA_FILE_SEXUAL_MUT.write("%s\n" % child.weight)
            else:
                DATA_FILE_ASEXUAL_MUT.write("%s\n" % child.weight)
        return child


class Population(object):
    def __init__(self, size, nMale, hasSexes, genLen, mutP, tiP, selStrength,
        posBaseWeights, bornLag, children_number, recFreq, recRate,
        sexSelStrength):
        # general parameters
        self.size = size
        self.nMale = nMale
        self.hasSexes = hasSexes
        self.genLen = genLen
        # moran step parameters
        self.mutP = mutP
        self.tiP = tiP
        self.selStrength = selStrength
        self.posBaseWeights = copy.deepcopy(posBaseWeights)
        self.bornLag = bornLag
        self.children_number = children_number
        self.recFreq = recFreq
        self.recRate = recRate
        self.sexSelStrength = sexSelStrength
        # init organisms
        self.organisms = [ Organism(generate_rand_genome(genLen), 'm')
            for _ in range(nMale) ] # create males
        self.organisms += [ Organism(generate_rand_genome(genLen), 'f')
            for _ in range(size-nMale) ] # add females
        random.shuffle(self.organisms)
        for age, org in enumerate(self.organisms):
            org.age = age
            org.set_weight(self.posBaseWeights)
        self.organisms.sort(key=lambda org: org.weight)

    def moran_step(self):
        assert ((self.size - self.nMale > self.bornLag) or
            (self.size > self.bornLag and not self.hasSexes))
        assert ((self.recFreq > 0 and self.recRate > 0) or
            (self.recFreq == 0 and self.recRate == 0))
        assert self.children_number > 0
        for org in self.organisms:
            org.age += 1
            org.youngestChildAge += 1
        dieInd = max(enumerate(self.organisms),
            key=lambda ind_org: ind_org[1].age)[0] # oldest organism will die
        descendantSex = self.organisms[dieInd].sex
        sexualReprod = random.random() < self.recFreq and self.recRate > 0
        if self.hasSexes:
            if sexualReprod:
                readyToSpawnOrgs = [ org for org in self.organisms
                    if org.sex=='m' or org.youngestChildAge>self.bornLag ]
            else:
                readyToSpawnOrgs = [ org for org in self.organisms
                    if org.sex=='f' and org.youngestChildAge>self.bornLag ]
        else:
            readyToSpawnOrgs = [ org for org in self.organisms
                if org.youngestChildAge>self.bornLag ]
        parentLowerInd = int((len(readyToSpawnOrgs)-1)*self.selStrength)
        parentInd = random.randint(parentLowerInd, len(readyToSpawnOrgs)-1)
        parent = readyToSpawnOrgs.pop(parentInd)
        parent2 = None
        if sexualReprod:
            if self.hasSexes:
                potential2ndParents = [ org for org in readyToSpawnOrgs
                    if org.sex != parent.sex ]
            else:
                potential2ndParents = readyToSpawnOrgs
            parent2LowerInd = int(self.sexSelStrength*
                (len(potential2ndParents)-1))
            parent2 = random.choice(potential2ndParents[parent2LowerInd:])
        descendant = parent.get_child(descendantSex, self.mutP, self.tiP,
            self.posBaseWeights, self.recRate, parent2)
        for _ in range(self.children_number - 1):
            another_descendant = parent.get_child(descendantSex, self.mutP,
                self.tiP, self.posBaseWeights, self.recRate, parent2)
            if another_descendant.weight > descendant.weight:
                descendant = another_descendant

        if RECORD:
            if self.recFreq == 1:
                DATA_FILE_SEXUAL.write("%s\n" % descendant.weight)
            else:
                DATA_FILE_ASEXUAL.write("%s\n" % descendant.weight)

        self.organisms.pop(dieInd)
        descendant_ind = bisect.bisect([org.weight for org in self.organisms],
            descendant.weight)
        self.organisms.insert(descendant_ind, descendant)

    def get_average_weight(self):
        return sum(o.weight for o in self.organisms)/(self.size*self.genLen)

    def get_base_freqs(self):
        freq_counter = collections.Counter(
            itertools.chain.from_iterable(o.genome for o in self.organisms) )
        totalSum = self.size*self.genLen
        return {base:count/totalSum for base, count in freq_counter.items()}

    def switch_pos_base_weights(self, posBaseWeights):
        self.posBaseWeights = copy.deepcopy(posBaseWeights)
        for org in self.organisms:
            org.set_weight(self.posBaseWeights)
        self.organisms.sort(key=lambda org: org.weight)


class Environment(object):
    def __init__(self, posBaseWeights1, posBaseWeights2, stepsBetweenSwitches,
        populations):
        self.posBaseWeights1 = posBaseWeights1
        self.posBaseWeights2 = posBaseWeights2
        self.stepsBetweenSwitches = stepsBetweenSwitches
        self.populations = populations

        self._lastSwitchStepsAgo = 0
        self._currentlyIn1 = True

    def trySwitch(self):
        if self._lastSwitchStepsAgo >= self.stepsBetweenSwitches:
            if self._currentlyIn1:
                for i, pop in enumerate(self.populations):
                    pop.switch_pos_base_weights(self.posBaseWeights2[i])
                self._currentlyIn1 = False
            else:
                for i, pop in enumerate(self.populations):
                    pop.switch_pos_base_weights(self.posBaseWeights1[i])
                self._currentlyIn1 = True
            self._lastSwitchStepsAgo = 0
            return True
        else:
            self._lastSwitchStepsAgo += 1
            return False


def run_animation(args):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots( nrows=len(args.pop_names)+1, ncols=1,
        figsize=(8,4*len(args.pop_names)), sharex=True, facecolor = 'w' )
    ax[-1].set_ylim(0, 1)
    ax[-1].set_xlim(0, 100)
    ax[-1].set_xlabel("iteration")
    ax[-1].set_ylabel("average weight")
    ax[-1].grid(True)
    ax[-1].set_title('weights')

    populations = []
    lines = []
    xdata = []
    lines_ydata = [[] for _ in args.pop_names]
    labels = []
    switch_line_xdata = []
    switch_line_ydata = []
    switch_line, = ax[-1].plot([], [], marker='^', linestyle=' ', color='k')

    freq_ydata = []
    freq_lines = []

    colors = 'brgymc'
    assert len(colors) >= len(args.pop_names)
    print("Number of iterations: %s" % args.iterations)
    print("Selection creterion switches: %s" % args.steps_between_switches)
    for j, pn in enumerate(args.pop_names):
        print("Generating population: %s" % pn)
        print("    Pop size:    %s" % args.pop_size[j])
        print("    Male num:    %s" % args.male_number[j])
        print("    Has sexes:   %s" % args.has_sexes[j])
        print("    Gen len:     %s" % args.genome_len[j])
        print("    Mut prob:    %s" % args.mut_prob[j])
        print("    Ti prob:     %s" % args.ti_prob[j])
        print("    Sel str:     %s" % args.sel_strength[j])
        print("    Born lag:    %s" % args.born_lag[j])
        print("    Children:    %s" % args.children_number[j])
        print("    Rec freq:    %s" % args.rec_freq[j])
        print("    Rec rate:    %s" % args.rec_rate[j])
        print("    Sex sel str: %s" % args.sex_sel_strength[j])
        print("    Pos weights: %s ..." %
            ' ... ; '.join(('%s: %s' % (k, ', '.join(map(str, v[:3])))
                           for k, v in args.pos_base_weights[j].items()))
             )
        print("    Alt weights: %s ..." %
            ' ... ; '.join(('%s: %s' % (k, ', '.join(map(str, v[:3])))
                           for k, v in args.alt_pos_base_weights[j].items()))
             )
        pop = Population(args.pop_size[j],
                         args.male_number[j],
                         args.has_sexes[j],
                         args.genome_len[j],
                         args.mut_prob[j],
                         args.ti_prob[j],
                         args.sel_strength[j],
                         args.pos_base_weights[j],
                         args.born_lag[j],
                         args.children_number[j],
                         args.rec_freq[j],
                         args.rec_rate[j],
                         args.sex_sel_strength[j])
        populations.append(pop)
        line, = ax[-1].plot([], [], lw=1, color=colors[j])
        lines.append(line)
        labels.append(pn)



        ax[j].set_ylim(0, 1)
        ax[j].set_xlim(0, 100)
        ax[-2].set_ylabel("freq")
        ax[j].grid(True)
       #~ ax[j].set_title("%s base frequencies" % pn)
    
        ax[j].set_axis_bgcolor(colors[j])
        ax[j].patch.set_alpha(0.15)
        freq_ydata.append([[],[],[],[]])
        line1, = ax[j].plot([], [], lw=1, color='r')
        line2, = ax[j].plot([], [], lw=1, color='y')
        line3, = ax[j].plot([], [], lw=1, color='b')
        line4, = ax[j].plot([], [], lw=1, color='g')
        freq_lines.append([line1, line2, line3, line4])

    ax[0].set_title("Populations nucleotides frequencies")



    env = Environment(args.pos_base_weights, args.alt_pos_base_weights,
        args.steps_between_switches, populations)
    fig.legend(lines, labels, loc='lower left', ncol=4)

    iStart = 0
    def data_gen():
        iCurr = iStart
        for iCurr in range(iStart, iStart+args.iterations):
            switch = env.trySwitch()
            for j, pop in enumerate(populations):
                pop.moran_step()
            iCurr += 1
            yield (iCurr,
                   [pop.get_average_weight() for pop in populations],
                   [pop.get_base_freqs() for pop in populations],
                   switch)

    ## TODO: add init_func for FuncAnimation

    def run(data):
        # update the data
        iteration, weights, freqs, switch = data
        if xdata == [] or xdata[-1] < iteration:
            xdata.append(iteration)
            for ydata, w in zip(lines_ydata, weights):
                ydata.append(w)

            xmin, xmax = ax[-1].get_xlim()
            if iteration >= xmax:
                for i in range(len(populations)+1):
                    ax[i].set_xlim( iStart,
                        min(1.5*xmax, iStart+args.iterations) )
                ax[-1].figure.canvas.draw()

            for line, ydata in zip(lines, lines_ydata):
                line.set_data(xdata, ydata)

            for i, freq_dict in enumerate(freqs):
                for j, b in enumerate(BASES):
                    freq_ydata[i][j].append(freq_dict[b])
                    if len(xdata) != len(freq_ydata[i][j]):
                        print(freq_dict)
                        print(freq_ydata)
                        sys.exit(1)
                    freq_lines[i][j].set_data(xdata, freq_ydata[i][j])

        if switch:
            switch_line_xdata.append(iteration)
            switch_line_ydata.append(10)
        switch_line.set_data(switch_line_xdata, switch_line_ydata)

        return lines + [switch_line] + \
            [l for pop_lines in freq_lines for l in pop_lines]

    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=20,
        repeat=False)
    plt.show()

    
def parse_pos_base_weights(arg):
    split_bases_weights = arg.split('_')
    assert len(split_bases_weights) == 4
    pos_base_weights = {}
    for base_weights in split_bases_weights:
        base, weights = base_weights.split(':')
        assert base in BASES
        pos_base_weights[base] = ast.literal_eval(weights)
    assert set(pos_base_weights.keys()) == set(BASES)
    return pos_base_weights


## TODO: add check of arguments correctness


def fit_pos_base_weights_to_gen_len(args):
    fitted_pos_base_weights = []
    gl = args.genome_len
    e = 1.e-6
    for j, pbw in enumerate(args.pos_base_weights):
        fitted = { b : w*(gl[j]//len(w))+w[:gl[j]%len(w)]
            for b, w in pbw.items() }
        assert all([ 1 - e < sum(fitted[b][i] for b in BASES) < 1 + e
            for i in range(gl[j]) ])
        fitted_pos_base_weights.append(fitted)
    args.pos_base_weights = fitted_pos_base_weights

    fitted_pos_base_weights = []
    for j, pbw in enumerate(args.alt_pos_base_weights):
        fitted = { b : w*(gl[j]//len(w))+w[:gl[j]%len(w)]
            for b, w in pbw.items() }
        assert all([ 1 - e < sum(fitted[b][i] for b in BASES) < 1 + e
            for i in range(gl[j]) ])
        fitted_pos_base_weights.append(fitted)
    args.alt_pos_base_weights = fitted_pos_base_weights


def check_arg_len_consistency(args):
    attr = ('pop_size', 'male_number', 'has_sexes', 'genome_len', 'mut_prob',
        'ti_prob', 'pos_base_weights', 'sel_strength', 'born_lag', 'rec_freq',
        'rec_rate', 'sex_sel_strength', 'alt_pos_base_weights',
        'children_number')
    attr_len_dict = {a:len(getattr(args, a)) for a in attr}
    popN = len(args.pop_names)
    if popN == 1:
        assert all(l==1 for l in attr_len_dict.values())
    else:
        assert all(l==popN or l==1 for l in attr_len_dict.values())
        for a,l in attr_len_dict.items():
            if l == 1:
                setattr(args, a, getattr(args, a)*popN)


def main():
    parser = argparse.ArgumentParser(description="Run evolution")
    parser.add_argument("-pn", "--pop-names",
                        nargs='*',
                        default=['population'])
    parser.add_argument("-ps", "--pop-size",
                        nargs='*',
                        type=int,
                        default=[100])
    parser.add_argument("-mn", "--male-number",
                        nargs='*',
                        type=int,
                        default=[50])
    parser.add_argument("-hs", "--has-sexes",
                        nargs='*',
                        type=ast.literal_eval,
                        default=[False])
    parser.add_argument("-gl", "--genome-len",
                        nargs='*',
                        type=int, 
                        default=[100])
    parser.add_argument("-mp", "--mut-prob",
                        nargs='*',
                        type=float,
                        default=[0.05])
    parser.add_argument("-tp", "--ti-prob",
                        nargs='*',
                        type=float,
                        default=[0.66])
    # Example of input: A:[0.8,0.5]_G:[0.2,0.3]_C:[0.0,0.1]_T:[0.0,0.1]
    parser.add_argument("-pbw", "--pos-base-weights",
                        nargs='*',
                        type=parse_pos_base_weights,
                        default=[{'A':[0.7],'C':[0.05],'T':[0],'G':[0.25]}])
            #~ default=[{'A':[0.7],'C':[0.1],'T':[0.1],'G':[0.1]}])
    parser.add_argument("-apbw", "--alt-pos-base-weights",
                        nargs='*',
                        type=parse_pos_base_weights,
                        default=[{'A':[0.3],'C':[0.2],'T':[0.1],'G':[0.4]}])
    parser.add_argument("-ss", "--sel-strength",
                        nargs='*',
                        type=float,
                        default=[0.5])
    parser.add_argument("-bl", "--born-lag",
                        nargs='*',
                        type=int,
                        default=[0])
    parser.add_argument("-cn", "--children-number",
                        nargs='*',
                        type=int,
                        default=[1])
    parser.add_argument("-rf", "--rec-freq",
                        nargs='*',
                        type=float,
                        default=[0])
    parser.add_argument("-rr", "--rec-rate",
                        nargs='*',
                        type=float,
                        default=[0])
    parser.add_argument("-sss", "--sex-sel-strength",
                        nargs='*',
                        type=float,
                        default=[0])
    parser.add_argument("-sbs", "--steps-between-switches",
                        type=int,
                        default=2500)
    parser.add_argument("-i", "--iterations",
                        type=int,
                        default=10000)

    args = parser.parse_args()
    check_arg_len_consistency(args)
    fit_pos_base_weights_to_gen_len(args)

    run_animation(args)



if __name__ == "__main__":
# python evol.py -pn asexual sexual -gl 100 -mp 0.05 0.12 -bl 0 -cn 4 -pbw A:[0.8,0.5]_G:[0.2,0.3]_C:[0.0,0.1]_T:[0.0,0.1] -rf 0 1 -rr 0 0.1 -hs 0 0 -ss 0 -sss 0 -sbs 45000 -i 40000

    RECORD = False

    if RECORD:
        DATA_FILE_SEXUAL = open("data/weights_sexual.txt", 'w')
        DATA_FILE_SEXUAL_REC = open("data/weights_sexual_rec.txt", 'w')
        DATA_FILE_SEXUAL_MUT = open("data/weights_sexual_mut.txt", 'w')
        DATA_FILE_ASEXUAL = open("data/weights_asexual.txt", 'w')
        DATA_FILE_ASEXUAL_MUT = open("data/weights_asexual_mut.txt", 'w')

    main()

    if RECORD:
        DATA_FILE_SEXUAL.close()
        DATA_FILE_SEXUAL_REC.close()
        DATA_FILE_SEXUAL_MUT.close()
        DATA_FILE_ASEXUAL.close()
        DATA_FILE_ASEXUAL_MUT.close()
'''
Launch example:

python evol.py -pn asexual sexual sexualsel -gl 200 -mp 0.1 -bl 10 -cn 3 -pbw A:[0.8,0.5]_G:[0.2,0.3]_C:[0.0,0.1]_T:[0.0,0.1] -apbw A:[0.2,0.3]_G:[0.8,0.5]_C:[0.0,0.1]_T:[0.0,0.1] -rf 0 1 1 -rr 0 0.1 0.1 -hs 0 1 1 -ss 0.6 -sss 0 0 0.6 -sbs 2000 -i 6000



Description.

Notation:
* and # - females
#       - females with child younger than born lag (not ready to born)
*       - females with younger child older than born lag (ready to born)
+       - males (not affected by born lag)

Population of given size (size) is generated with defined number of males
(male_number), thus the number of females = size - male_number.


Moran evolution step:

|    min weight               <= weight <=               max weight
|    
|    *+**++*+**+++****+*#+++**+**##*+++*++*#++++#**+*+**++##+#*#++*
|    |-------------------------   size   -------------------------|
V    
|    organisms ready to spawn (S), # females are filtered out
|
|    *+**++*+**+++****+*+++**+***+++*++*++++**+*+**+++*++*
|    |----------------------   S   ----------------------|
V                           |- (1-selection_strength)*S -|
|
|                 first parent is selected from (1-selection_strength)*S
|                 organisms with best weights randomly. Assuming that 1st
|                 parent is male, recombination frequency is 1 and our
V                 population has has_sexes parameter == True, the second
|                 parent should be selected from females which are ready
|                 to born (B)
|
|    *************************
V    |--------   B   --------| 
|      |- (1-sex_sel_str)*B -|   
|    
|    second parent is selected from (1-sex_sel_str)*B
|    organisms with best weights randomly.
V    
|    Oldest organism in the population dies.
|    New organism is generated from the selected parents and inserted into
|    the population according to its weight (if children_number parameter
|    is > 1 than children_number descendants are generated but only the one
V    with the largest weight goes further). New organism has the same sex as
|    the oldest organism vanished at this step, so the ratio of sexes is
V    always constant.

Next step


Parameters:

rec_freq - recombination frequency, probability that sexual reproduction will
happen in the step. 0 - pure parthenogenesis, 1 - pure sexual reproduction,
0 < rec_freq < 1 - recombination happens occasionally

has_sexes - determines whether sexes are taken into account. If False, all
organisms are treated equally (each organism can be mated with any other and
has a born lag). If true, sexes are different: males are not affected by born
lag and cannot produce a child without females on the other hand females are
affected by the born lag and can produce the child alone if parthenogenesis
will happen in the step

rec_rate - determines probability of crossover events in recombination. If
rec_rate is 1 all adjacent bases in child's genome will be from different
parents

It is pointless to have rec_rate > 0 if rec_freq is 0 and vice versa
'''