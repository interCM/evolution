from __future__ import division
import random
import bisect
import collections
import itertools
import copy
import ast
import argparse
import sys
import numpy
import itertools
from math import log

BASES = "AGCT"
TRANSITIONS = {'A':'G','G':'A','C':'T','T':'C'}
TRANSVERSIONS = {'A':'CT','G':'CT','C':'AG','T':'AG'}

# These are the "Tableau 20" colors as RGB
TABLEAU20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1], which is the format matplotlib accepts
for i in range(len(TABLEAU20)):  
    r, g, b = TABLEAU20[i]  
    TABLEAU20[i] = (r/255., g/255., b/255.)

def get_gi(freqs):
    gi = 2
    for fr in freqs:
        if fr > 0:
            gi += fr*log(fr,2)
    return gi


def generate_rand_genome(genLen):
    return [ random.choice(BASES) for _ in range(genLen) ]


def mutate_base(base, tiP):
    if random.random() < tiP:
        return TRANSITIONS[base]
    else:
        return random.choice(TRANSVERSIONS[base])


def parentIndexGenerator(recRate):
    ind = random.randint(0, 1)
    while True:
        ind = (ind+1)%2 if random.random() < recRate else ind
        yield ind


def write_info_dict(info_dict, f):
    for k, v in info_dict.iteritems():
        if isinstance(v, list):
            for e in v:
                f.write("%s\t%s\n" % (k, e))
        else:
            f.write("%s\t%s\n" % (k, v))


class Organism(object):
    def __init__(self, genome, sex, genome_origin=None):
        self.genome = genome
        self.sex = sex
        self.age = 0
        self.weight = 1
        self.youngestChildAge = 1000000
        self.nChildren = 0
        self.nMut = 0
        self.genome_origin = ( genome[:] if genome_origin is None
            else genome_origin )

    def set_weight(self, posBaseWeights):
        self.weight = sum(posBaseWeights[b][i]
                for i,b in enumerate(self.genome))

    # Modifies genome sequence with respect to the given probabilities of
    # mutation and transition.
    # Sets nMut equal to the number of occurred mutations 
    def mutate(self, mutP, tiP):
        n = len(self.genome)
        mut_pos_ind = numpy.arange(n)[numpy.random.rand(n) < mutP]
        self.nMut = len(mut_pos_ind)
        for i in mut_pos_ind:
            self.genome[i] = mutate_base(self.genome[i],tiP)

    def recombine(self, org, recRate):
        parInd = parentIndexGenerator(recRate)
        ind = [next(parInd) for i in xrange(len(self.genome))]
        tmp_zip = zip(self.genome, org.genome)
        self.genome = [tmp_zip[i] for i in ind]
        tmp_zip = zip(self.genome_origin, org.genome_origin)
        self.genome_origin = [tmp_zip[i] for i in ind]

    # Returns:
    # child - new Organism instance
    # Modifies:
    # info_dict - a dictionary with mutation and recombination statistics
    def get_child(self, sex, mutP, tiP, posBaseWeights, info_dict,
        recRate=0, secondParentOrg=None):
        self.youngestChildAge = 0
        self.nChildren += 1
        child = Organism(self.genome[:], sex, self.genome_origin[:])
        if not secondParentOrg is None:
            child.recombine(secondParentOrg, recRate)
            secondParentOrg.nChildren += 1
            child.set_weight(posBaseWeights)
            info_dict["after_rec_weight"].append(child.weight)
        child.mutate(mutP, tiP)
        child.set_weight(posBaseWeights)
        info_dict["after_mut_weight"].append(child.weight)
        return child

    # Returns:
    # number of mutations compared to origin
    def compare_to_origin(self):
        return sum(( 1 for g, o in zip(self.genome, self.genome_origin)
            if g != o ))


class Population(object):
    def __init__(self, size, nMale, hasSexes, genLen, mutP, tiP, selStrength,
        posBaseWeights, bornLag, children_number, recFreq, recRate,
        sexSelStrength, popId=None):
        # general parameters
        self.popId = popId
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
        # step_info_dict contains different information for recording
        # if None - nothing will be recorded
        # value in step_info_dict is a list than all list items will be
        # recorded consequently with the same tag - corresponding tag
        step_info_dict = collections.defaultdict(list)
        for org in self.organisms:
            org.age += 1
            org.youngestChildAge += 1
        dieInd = random.randint(0, self.size-1) # random organism will die
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
            self.posBaseWeights, step_info_dict, self.recRate, parent2)
        for _ in range(self.children_number - 1):
            another_descendant = parent.get_child(descendantSex, self.mutP,
                self.tiP, self.posBaseWeights, step_info_dict, self.recRate,
                parent2)
            if another_descendant.weight > descendant.weight:
                descendant = another_descendant

        # record required parameters
        step_info_dict["after_sel_weight"] = descendant.weight
        step_info_dict["average_weight"] = self.get_average_weight()
        step_info_dict["n_mut"] = descendant.nMut
        step_info_dict["base_freqs"] = self.get_base_freqs()
        step_info_dict["av_n_mut_origin"] = \
            self.get_av_mut_n_compared_to_origin()

        self.organisms.pop(dieInd)
        descendant_ind = bisect.bisect([org.weight for org in self.organisms],
            descendant.weight)
        self.organisms.insert(descendant_ind, descendant)
        return step_info_dict

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
        
    def get_av_mut_n_compared_to_origin(self):
        return sum(o.compare_to_origin() for o in self.organisms)/self.size
        
    def update_origin_genomes(self):
        for o in self.organisms:
            o.genome_origin = o.genome[:]


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
    from matplotlib.patches import Rectangle
    import matplotlib.animation as animation

    n_pop = len(args.pop_names)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

    # Remove the plot frame lines. They are unnecessary chartjunk.  
    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)  
        ax.spines["bottom"].set_visible(False)  
        ax.spines["right"].set_visible(False)  
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()

    lower_y = 0
    upper_y = 1
    bar_height = 0.1
    between_bars_step = (upper_y - lower_y)/(n_pop + 1.)
    bar_pos = [(1 + i)*between_bars_step - bar_height/2 for i in range(n_pop)]
    weight_line_pos = [(1 + i)*between_bars_step for i in range(n_pop)]

    ax1.set_yticks([bp + bar_height/2 for bp in bar_pos])
    ax1.set_yticklabels(args.pop_names, fontsize=14)
    ax1.set_xlabel("Normalized weight", fontsize=14)
    ax1.set_xlim((0, 1))
    ax1.set_ylim((lower_y, upper_y))

    ax2.set_yticks([bp + bar_height/2 for bp in bar_pos])
    ax2.set_yticklabels(args.pop_names, fontsize=14)
    ax2.set_xlabel("N mutations compared to origin", fontsize=14)
    upper_mut_x_lim = max(args.genome_len)
    ax2.set_xlim((0, upper_mut_x_lim))
    ax2.set_ylim((lower_y, upper_y))

    ax3.set_xlabel("Nucleotide frequencies", fontsize=14)
    ax3.set_xlim((0, 1))
    ax3.set_ylim((lower_y, upper_y))

    populations = []

    white_rect = Rectangle((0, 0), 1, upper_y, facecolor='white',
            edgecolor = "none")
    ax1.add_patch(white_rect)
    ax1_background = [white_rect] # contains background rect and grid lines
    white_rect = Rectangle((0, 0), upper_mut_x_lim, upper_y,
        facecolor='white', edgecolor = "none")
    ax2.add_patch(white_rect)
    ax2_background = [white_rect]

    # add gi, iteration and generations since last origin update labels
    iter_label = ax1.text(.015, 0.95, "0", fontsize=12,
        bbox=dict(facecolor='white', edgecolor = "none"))
    switch_label = ax1.text(.015, 0.9, "0", fontsize=12, color=TABLEAU20[6],
        bbox=dict(facecolor="white", edgecolor = "none"))
    gen_since_last_origin_update_label = ax2.text(upper_mut_x_lim//20, 0.95, "0",
        fontsize=12, bbox=dict(facecolor='white', edgecolor = "none"))
    counter_labels = [iter_label, switch_label,
        gen_since_last_origin_update_label]
    gi_labels = []
    for i in range(n_pop):
        gi_text = ax.text(.01, bar_pos[i] + .035, "0", fontsize=12)
        gi_labels.append(gi_text)

    # set ticks
    x_ticks = numpy.linspace(0,1,11)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(list(map(str, x_ticks)))
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(list(map(str, x_ticks)))

    ax2_x_ticks = numpy.arange(0, upper_mut_x_lim+1, 5)
    ax2.set_xticks(ax2_x_ticks)
    ax2_tick_labels = [str(t) if i%2==0 else ""
        for i, t in enumerate(ax2_x_ticks)]
    ax2.set_xticklabels(ax2_tick_labels)
    # ax2.set_xticklabels(list(map(str, ax2_x_ticks)))

    # plot vertical grid lines
    for x in x_ticks[1:-1]:
        l, = ax1.plot([x,x], [lower_y, upper_y], linestyle="--", lw=0.5,
            color='k', alpha=0.3)
        ax1_background.append(l)
        ax3.plot([x,x], [lower_y, upper_y], linestyle="--", lw=0.5, color='k',
            alpha=0.3)

    for x in ax2_x_ticks[1:-1]:
        l, = ax2.plot([x,x], [lower_y, upper_y], linestyle="--", lw=0.5,
            color='k', alpha=0.3)
        ax2_background.append(l)

    # init barplots
    weight_bars = ax1.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[5],
        edgecolor="none")
    mut_bars = ax2.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[5],
        edgecolor="none")
    weight_std_lines = []
    mut_std_lines = []
    for wlp in weight_line_pos:
        line, = ax1.plot([0, 0], [wlp, wlp], color='k', lw=1, marker='|',
            linestyle=':', markersize=8)
        weight_std_lines.append(line)
        line, = ax2.plot([0, 0], [wlp, wlp], color='k', lw=1, marker='|',
            linestyle=':', markersize=8)
        mut_std_lines.append(line)
    t_freq_bars = ax3.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[0],
        edgecolor = "none")
    c_freq_bars = ax3.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[1],
        edgecolor = "none")
    g_freq_bars = ax3.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[2],
        edgecolor = "none")
    a_freq_bars = ax3.barh(bar_pos, [0]*n_pop, bar_height, color=TABLEAU20[3],
        edgecolor = "none")
        
    # common parameters for all populations:
    print("Update origin genomes: %s" %
        ", ".join(map(str, args.update_origin_generations)))
    
    # init populations
    pop_weights, compared_to_origin_muts = [], []
    for j, pn in enumerate(args.pop_names):
        pop_weights.append(collections.deque([], 1000))
        compared_to_origin_muts.append(collections.deque([], 1000))

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
                         args.sex_sel_strength[j],
                         pn)
        populations.append(pop)
    
    env = Environment(args.pos_base_weights, args.alt_pos_base_weights,
        args.steps_between_switches, populations)

    def data_gen():
        iStart = 0
        iSwitch = 0
        iSinceLastOriginUpdate = 0
        for iCurr in range(iStart, iStart+args.iterations):
            update_origin_genomes = iCurr in args.update_origin_generations
            if update_origin_genomes:
                iSinceLastOriginUpdate = 0
            else:
                iSinceLastOriginUpdate += 1
            switch = env.trySwitch()
            iSwitch = 0 if switch else iSwitch + 1
            average_weight, base_freqs, av_n_mut_origin = [], [], []
            for i, pop in enumerate(populations):
                if update_origin_genomes:
                    pop.update_origin_genomes()
                    compared_to_origin_muts[i].clear()
                step_info_dict = pop.moran_step()
                average_weight.append(step_info_dict["average_weight"])
                base_freqs.append(step_info_dict["base_freqs"])
                av_n_mut_origin.append(step_info_dict["av_n_mut_origin"])
                if args.record:
                    write_info_dict(step_info_dict, POP_FILE_DICT[pop.popId])
            yield (iCurr+1, average_weight, base_freqs, av_n_mut_origin,
                iSinceLastOriginUpdate, iSwitch)

    #TODO: add init_func for FuncAnimation

    def run(data):
        # update the data
        iteration, weights, freqs, av_n_mut_origin, iOrUpdate, iSwitch = data
        for i, (pw, nm) in enumerate(zip(weights, av_n_mut_origin)):
            pop_weights[i].append(pw)
            compared_to_origin_muts[i].append(nm)
            weight_bars[i].set_width(pw)
            # mut_bars[i].set_width(nm)
            mut_bars[i].set_width(numpy.mean(compared_to_origin_muts[i]))

        for i, (y, wl, ml) in enumerate(zip(weight_line_pos, weight_std_lines,
            mut_std_lines)):
            min_w = min(pop_weights[i])
            max_w = max(pop_weights[i])
            wl.set_data([min_w, max_w], [y, y])
            min_m = min(compared_to_origin_muts[i])
            max_m = max(compared_to_origin_muts[i])
            ml.set_data([min_m, max_m], [y, y])

        a_freqs = numpy.array([fq['A'] for fq in freqs])
        g_freqs = numpy.array([fq['G'] for fq in freqs]) + a_freqs
        c_freqs = numpy.array([fq['C'] for fq in freqs]) + g_freqs
        t_freqs = numpy.array([fq['T'] for fq in freqs]) + c_freqs

        for fb, f in zip(a_freq_bars, a_freqs):
            fb.set_width(f)

        for fb, f in zip(g_freq_bars, g_freqs):
            fb.set_width(f)

        for fb, f in zip(c_freq_bars, c_freqs):
            fb.set_width(f)

        for fb, f in zip(t_freq_bars, t_freqs):
            fb.set_width(f)

        for gi_text, fr_dict in zip(gi_labels, freqs):
            gi = get_gi(fr_dict.values())
            gi_text.set_text("%.3f" % gi)

        iter_label.set_text("%s" % iteration)
        switch_label.set_text("%s" % iSwitch)
        gen_since_last_origin_update_label.set_text("%s" % iOrUpdate)

        to_redraw_list = itertools.chain(ax1_background, ax2_background,
            counter_labels, weight_bars, mut_bars, weight_std_lines,
            mut_std_lines, t_freq_bars, c_freq_bars, g_freq_bars, a_freq_bars,
            gi_labels)
        return to_redraw_list


    ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
        repeat=False)
    plt.tight_layout()
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
    parser.add_argument("-uog", "--update-origin-generations",
                        nargs='*',
                        type=int,
                        default=[0])
    parser.add_argument("-sbs", "--steps-between-switches",
                        type=int,
                        default=2500)
    parser.add_argument("-i", "--iterations",
                        type=int,
                        default=10000)

    parser.add_argument("--record",
                        action="store_true")

    args = parser.parse_args()
    check_arg_len_consistency(args)
    fit_pos_base_weights_to_gen_len(args)

    if args.record:
        # if record is required, check whether all necessary files are given
        assert set(POP_FILE_DICT.keys()) >= set(args.pop_names)
        for k, v in POP_FILE_DICT.items():
            POP_FILE_DICT[k] = open(v, 'w')
        
    run_animation(args)

    if args.record:
        for f in POP_FILE_DICT.values():
            f.close()



if __name__ == "__main__":
# figure3:  python evol.py -pn asex sex -gl 100 -mp 0.0152 0.05 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 1 -rr 0 0.1 -hs 0 1 -ss 0 -sss 0 0 -sbs 55000 -i 50000
# figure3v2: python evol.py -pn asexlow asexhigh sex -gl 100 -mp 0.0152 0.05 0.05 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 0 1 -rr 0 0 0.1 -hs 0 0 1 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure2:  python evol.py -pn asex005 asex01 sex005 sex01 -ps 100 -gl 100 -mp 0.05 0.1 0.05 0.1 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 0 1 1 -rr 0 0 0.1 0.1 -hs 0 0 1 1 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure2v2: repeat experiment according to the figure
# figure1s: python evol.py -pn asex2 asex3 sex2 sex3 -gl 100 -mp 0.05 -bl 0 -cn 2 3 2 3 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 0 1 1 -rr 0 0 0.1 0.1 -hs 0 0 1 1 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure2sA: python evol.py -pn asex100 asex200 asex400 asex800 -gl 100 200 400 800 -mp 0.02 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 -rr 0 -hs 0 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure2sS: python evol.py -pn sex100 sex200 sex400 sex800 -gl 100 200 400 800 -mp 0.02 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 1 -rr 0.1 -hs 1 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure3s: python evol.py -pn asex sex sexsel -gl 100 -mp 0.05 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0 1 1 -rr 0 0.1 0.1 -hs 0 1 1 -ss 0 -sss 0 0 0.5 -sbs 55000 -i 50000
# figure4s: python evol.py -pn sex002 sex005 sex01 sex05 sex1 -gl 100 -mp 0.05 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 1 -rr 0.02 0.05 0.1 0.5 1.0 -hs 1 -ss 0 -sss 0 -sbs 55000 -i 50000
# figure5s: python evol.py -pn sex005 sex01 sex02 sex05 sex1 -gl 100 -mp 0.05 -bl 0 -cn 4 -pbw A:[0.65]_G:[0.25]_C:[0.1]_T:[0.0] -rf 0.05 0.1 0.2 0.5 1 -rr 0.1 -hs 1 -ss 0 -sss 0 -sbs 55000 -i 50000
    
    # keys should correspond to population ids
    POP_FILE_DICT = {"asexlow":"data/fig3_asexlow_stats.txt",
                     "asexhigh":"data/fig3_asexhigh_stats.txt",
                     "sex":"data/fig3_sex_stats.txt"}

    main()


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