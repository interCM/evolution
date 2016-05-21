import sys
assert sys.version_info[:2] == (3, 5), "Use python version 3.5"
import os, random, bisect, math, ast, configparser, shutil, itertools
from collections import Counter
from operator import attrgetter
import numpy as np
import monitor


BASES = "AGCT" # nucleotides
TR = {'A':'G','G':'A','C':'T','T':'C'}
TV = {'A':'CT','G':'CT','C':'AG','T':'AG'}

def get_gi(freqs):
    """ Calculate GI for given nucleotide frequencies.
    Args:
    freqs = [fA, fG, fC, fT] - list of nucleotide frequencies, sum(freqs)=1
    Return:
    GI = -(fA*log2(fA) + fG*log2(fG) + fC*log2(fC) + fT*log2(fT))
    """
    gi = 2.
    for fr in freqs:
        if fr > 0:
            gi += fr*math.log2(fr)
    return gi

def generate_random_genome(L):
    """ Create random list of nucleotides (genome).
    Args:
    L - length og genome
    Return:
    Random nucleotide list of length L
    """
    return [random.choice(BASES) for _ in range(L)]


class AsexualOrganism(object):
    def __init__(self, genome, pbw, weight=None, origin=None, age=0):
        """ Represents an organism without sex. Depending on population
        parameters such organisms can reproduce without partner or mate with
        any other organism in the population regardless of its sex.
        Args:
        genome - list of nucleotides
        pbw    - positional base weights. List of dictionaries, each dict
                 contains nucleotide weights for a corresponding position.
                 If len(pbw) < len(genome), then pbw will be repeated several
                 times until it covers all positions in the genome.
        weight - weight of the organism (float or None). If None, it will be
                 calculated according to the given pbw.
        origin - genome used as an origin reference for this organism
        age    - age of the organism (int)
        """
        self.genome = genome
        cpbw = itertools.cycle(pbw)
        self.pbw = [next(cpbw) for _ in range(len(self.genome))]
        self.set_weight(weight)
        self.set_origin(origin)
        self.age = age

    def set_weight(self, weight=None):
        """ Set organism's weight.
        Args:
        weight - weight of the organism (float or None). If None, it will be
                 calculated according to the self.pbw
        """
        self.weight = ( sum(self.pbw[i][b] for i, b in enumerate(self.genome))
            if weight is None else weight )

    def set_origin(self, origin=None):
        """ Set origin genome for the organism.
        Args:
        origin - list of nucleotide or None. If None, current organism's
                 genome wil be set as origin
        """
        self.origin = self.genome[:] if origin is None else origin

    def get_weight_per_position(self):
        """ Calculate organism's average positional weight.
        Return:
        average weight per positinon (float)
        """
        return self.weight/len(self.genome)

    def compare_to_origin(self):
        """ Calculate organism's average number of mutations per positinon as
        compared to origin genome.
        Return:
        average number of mutations per positinon (float)
        """
        n_mut = sum((1 for g, o in zip(self.genome, self.origin) if g != o))
        return n_mut/len(self.genome)

    def update_pbw(self, new_pbw):
        """ Change organism's positional base weights.
        """
        cpbw = itertools.cycle(new_pbw)
        pbw = [next(cpbw) for _ in range(len(self.genome))]
        self.pbw = pbw
        self.set_weight()

    def mutate(self, mP, tiP):
        """ Introduce mutations to the organism's genome.
        Args:
        mP  - probability of mutation per positinon (float <= 1)
        tiP - probability thet occured mutation will be a transition
              (0 <= float <= 1)
        """
        mut_pos = np.flatnonzero(np.random.rand(len(self.genome)) < mP)
        ti_p = np.random.rand(len(mut_pos))
        for i, p in enumerate(mut_pos):
            b = self.genome[p]
            nb = TR[b] if ti_p[i] < tiP else random.choice(TV[b])
            self.genome[p] = nb
            self.weight += self.pbw[i][nb] - self.pbw[i][b]

    def recombine(self, other, rr):
        """ Recombine organism's genome with genome of another organism.
        Args:
        other - another organism
        rr    - probability of crossover event per base (0 <= float <= 1)
        """
        rec_pos = list(np.flatnonzero(np.random.rand(len(self.genome)) < rr))
        rec_pos.append(len(self.genome))
        rec_pos.insert(0, 0)
        intervals = list(zip(rec_pos[:-1], rec_pos[1:]))
        start = 0 if random.random() < 0.5 else 1
        for s, e in intervals[start::2]:
            self.weight += sum( self.pbw[s+i][nb] - self.pbw[s+i][b]
                for i, (b, nb) in
                    enumerate(zip(self.genome[s:e], other.genome[s:e])) )
            self.genome[s:e] = other.genome[s:e]
            self.origin[s:e] = other.origin[s:e]

    @classmethod
    def generate_random(cls, genome_len, pbw):
        """ Generate a random organism with a given lenrth of genome and
        positional base weights.
        Args:
        genome_len - length of genome (int)
        pbw        - positional base weights (list of dicts)
        Return:
        AsexualOrganism
        """
        genome = generate_random_genome(genome_len)
        return cls(genome, pbw)

    def get_child(self, mP, tiP, parent2, rr):
        """ Produce a descendant.
        Args:
        mP      - probability of mutation per positinon (float <= 1)
        tiP     - probability thet occured mutation will be a transition
                  (0 <= float <= 1)
        parent2 - mating partner (organism or None)
        rr      - probability of crossover event per base (0 <= float <= 1)
        Return:
        descendant AsexualOrganism
        """
        child = AsexualOrganism( self.genome[:], self.pbw, self.weight,
            self.origin[:] )
        if not parent2 is None:
            child.recombine(parent2, rr)
            MONITOR.write("WEIGHT_AFTER_REC", child.weight/len(child.genome))
        child.mutate(mP, tiP)
        MONITOR.write("WEIGHT_AFTER_MUT", child.weight/len(child.genome))
        return child


class SexualOrganism(AsexualOrganism):
    def __init__(self, genome, pbw, sex, weight=None, origin=None, age=0):
        """ Represents an organism with sex. Depending on population
        parameters such organisms can reproduce without partner or mate with
        other organism in the population having different sex.
        Args:
        same as in AsexualOrganism
        sex - sex of the organism (str, 'M' or 'F')
        """
        super().__init__(genome, pbw, weight, origin, age)
        self.sex = sex

    @classmethod
    def generate_random(cls, genome_len, pbw, sex):
        """ Generate a random organism with a given lenrth of genome,
        positional base weights and sex.
        Args:
        same as in AsexualOrganism.generate_random + sex
        Return:
        SexualOrganism
        """
        genome = generate_random_genome(genome_len)
        return cls(genome, pbw, sex)

    def get_child(self, sex, mP, tiP, parent2, rr):
        """ Produce a descendant.
        Args:
        same as in AsexualOrganism.get_child + sex
        Return:
        descendant SexualOrganism
        """
        child = SexualOrganism( self.genome[:], self.pbw, sex, self.weight,
            self.origin[:] )
        if not parent2 is None:
            child.recombine(parent2, rr)
            MONITOR.write("WEIGHT_AFTER_REC", child.weight/len(child.genome))
        child.mutate(mP, tiP)
        MONITOR.write("WEIGHT_AFTER_MUT", child.weight/len(child.genome))
        return child


class AsexualPopulation(object):
    def __init__(self, organisms, mut_prob, ti_prob, sel_strength,
            children_number, rec_freq, rec_rate, partner_sel_strength,
            eliminate_oldest=False):
        """ Represents a population os AsexualOrganisms.
        Args:
        organisms - list of AsexualOrganisms
        mut_prob  - probability of mutation per positinon (0 <= float <= 1) 
        ti_prob   - probability thet occured mutation will be a transition
                    (float <= 1)
        sel_strength    - selection strength, determines which fraction of
                          the populaiton is considered for reproduction at
                          each reproduction round (0 <= float <= 1)
        children_number - number of descendants generated per replication
                          round. Only a descendant with highest weight is
                          kept in the population, others are ignored (int)
        rec_freq  - probability of recombination event per round of
                    replication (float <= 1)
        rec_rate  - probability of crossover event per base (0 <= float <= 1)
        partner_sel_strength - determines which fraction of the populaiton is
                               considered when mating partner is selected
                               (0 <= float <= 1)
        eliminate_oldest     - boolean flag. If true an oldest organism is
                               eliminated from the population after each
                               reproduction round, otherwise random organism
                               is eliminated
        """
        self.organisms = organisms
        self.organisms.sort(key=lambda o: o.weight)
        self.mP = mut_prob
        self.tiP = ti_prob
        self.ss = sel_strength
        self.cn = children_number
        self.rr = rec_rate
        self.rf = rec_freq
        self.pss = partner_sel_strength
        self.eliminate_oldest = eliminate_oldest
        # initialize positional base frequencies
        gen_len = len(self.organisms[0].genome)
        self.pos_base_count = [{b:0 for b in BASES} for _ in range(gen_len)]
        for i,b in enumerate(zip(*map(attrgetter("genome"), self.organisms))):
            self.pos_base_count[i].update(Counter(b))

    def get_average_weight_per_position(self):
        w = sum(o.get_weight_per_position() for o in self.organisms)
        return w/len(self.organisms)

    def update_pbw(self, new_pbw):
        for o in self.organisms:
            o.update_pbw(new_pbw)

    def set_current_genome_as_origin(self):
        for o in self.organisms:
            o.set_origin()

    def update_pos_base_freq(self, offspring, died_org):
        for i, (bo, bd) in enumerate(zip(offspring.genome, died_org.genome)):
            self.pos_base_count[i][bo] += 1
            self.pos_base_count[i][bd] -= 1

    def get_average_gi_per_position(self):
        org_n = len(self.organisms)
        gi = 0
        for b_count in self.pos_base_count:
            freqs = [n/org_n for n in b_count.values()]
            gi += get_gi(freqs)
        return gi/len(self.pos_base_count)

    def get_average_mut_prob_compared_to_origin(self):
        size = len(self.organisms)
        return sum(o.compare_to_origin() for o in self.organisms)/size

    def get_org_to_die_ind(self):
        if self.eliminate_oldest:
            # org with maximum age will be eliminated from the population
            die_ind, _ = max( enumerate(self.organisms),
                key=lambda ind_org: ind_org[1].age )
        else:
            # random org will be eliminated from the population
            die_ind = random.randint(0, len(self.organisms)-1)
        return die_ind

    def get_parent(self, fertile, ss):
        p_lower = int((len(fertile)-1)*ss)
        p_index = random.randint(p_lower, len(fertile)-1)
        parent = fertile.pop(p_index)
        return parent

    def get_offspring(self, parent1, parent2):
        offspring = parent1.get_child(self.mP, self.tiP, parent2, self.rr)
        for _ in range(self.cn - 1):
            another_offspring = parent1.get_child( self.mP, self.tiP, parent2,
                self.rr )
            if another_offspring.weight > offspring.weight:
                offspring = another_offspring
        return offspring

    @classmethod
    def generate_random(cls, pop_size, mut_prob, ti_prob, sel_strength,
            children_number, rec_freq, rec_rate, partner_sel_strength,
            genome_len, pbw):
        organisms = [ AsexualOrganism.generate_random(genome_len, pbw)
            for _ in range(pop_size) ]
        return cls( organisms, mut_prob, ti_prob, sel_strength,
            children_number, rec_freq, rec_rate, partner_sel_strength )

    def moran_step(self):
        """ Moran reproduction round.
        """
        for o in self.organisms:
            o.age += 1
        die_ind = self.get_org_to_die_ind()
        fertile = self.organisms[:]
        parent1 = self.get_parent(fertile, self.ss)
        recombine = random.random() < self.rf
        parent2 = self.get_parent(fertile, self.pss) if recombine else None
        died_org = self.organisms.pop(die_ind)
        offspring = self.get_offspring(parent1, parent2)
        MONITOR.write( "WEIGHT_AFTER_SEL",
            offspring.weight/len(offspring.genome) )
        self.update_pos_base_freq(offspring, died_org)
        offspring_ind = bisect.bisect( [o.weight for o in self.organisms],
            offspring.weight )
        self.organisms.insert(offspring_ind, offspring)
        MONITOR.write("AV_WEIGHT", self.get_average_weight_per_position())
        MONITOR.write("AV_GI", self.get_average_gi_per_position())
        MONITOR.write( "AV_MUT",
            self.get_average_mut_prob_compared_to_origin() )



class SexualPopulation(AsexualPopulation):
    def get_offspring(self, parent1, parent2, sex):
        offspring = parent1.get_child( sex, self.mP, self.tiP, parent2,
            self.rr )
        for _ in range(self.cn - 1):
            another_offspring = parent1.get_child( sex, self.mP, self.tiP,
                parent2, self.rr )
            if another_offspring.weight > offspring.weight:
                offspring = another_offspring
        return offspring

    @classmethod
    def generate_random( cls, pop_size, mut_prob, ti_prob, sel_strength,
        children_number, rec_freq, rec_rate, partner_sel_strength,
        male_number, genome_len, pbw ):
        organisms = [ SexualOrganism.generate_random(genome_len, pbw, 'F')
            for _ in range(pop_size - male_number) ]
        organisms += [ SexualOrganism.generate_random(genome_len, pbw, 'M')
            for _ in range(male_number) ]
        return cls( organisms, mut_prob, ti_prob, sel_strength,
            children_number, rec_freq, rec_rate, partner_sel_strength )

    def moran_step(self):
        for o in self.organisms:
            o.age += 1
        die_ind = self.get_org_to_die_ind()
        fertile = self.organisms[:]
        parent1 = self.get_parent(fertile, self.ss)
        recombine = random.random() < self.rf
        if recombine:
            fertile = [o for o in fertile if o.sex != parent1.sex]
            parent2 = self.get_parent(fertile, self.pss)
        else:
            parent2 = None
        died_org = self.organisms.pop(die_ind)
        offspring = self.get_offspring(parent1, parent2, died_org.sex)
        MONITOR.write( "WEIGHT_AFTER_SEL",
            offspring.weight/len(offspring.genome) )
        self.update_pos_base_freq(offspring, died_org)
        offspring_ind = bisect.bisect( [o.weight for o in self.organisms],
            offspring.weight )
        self.organisms.insert(offspring_ind, offspring)
        MONITOR.write("AV_WEIGHT", self.get_average_weight_per_position())
        MONITOR.write("AV_GI", self.get_average_gi_per_position())
        MONITOR.write( "AV_MUT",
            self.get_average_mut_prob_compared_to_origin() )



class Environment(object):
    def __init__(self, populations, update_origin_at_iter,
        pos_base_weights_switch_at_iter, primary_pos_base_weights,
        alternative_pos_base_weights):
        self.populations = populations
        self.update_origin_at_iter = update_origin_at_iter
        self.pos_base_weights_switch_at_iter = pos_base_weights_switch_at_iter
        self.primary_pos_base_weights = primary_pos_base_weights
        self.alternative_pos_base_weights = alternative_pos_base_weights

    @classmethod
    def create_from_config(cls, config):
        e = config["environment"]
        uoai = process_int_list_cfg_entry(e["update_origin_at_iter"])
        ppbw = process_pbw_cfg_entry(e["primary_pos_base_weights"])
        assert not (ppbw is None)
        apbw = process_pbw_cfg_entry(e["alternative_pos_base_weights"])
        pbwsai = \
            process_int_list_cfg_entry(e["pos_base_weights_switch_at_iter"])
        assert not ( apbw is None and pbwsai != [] and
            min(pbwsai) < config.getint("common", "iterations") )
        pop_sec = [e for e in config.sections() if e.startswith("population")]
        populations = {}
        for p in pop_sec:
            curr_pop = config[p]
            pop_size = int(curr_pop["pop_size"])
            assert pop_size > 0
            genome_len = int(curr_pop["genome_len"])
            assert genome_len > 0
            mut_prob = float(curr_pop["mut_prob"])
            assert 0 <= mut_prob <= 1
            ti_prob = float(curr_pop["ti_prob"])
            assert 0 <= ti_prob <= 1
            sel_strength = float(curr_pop["sel_strength"])
            assert 0 <= sel_strength <= 1
            children_number = int(curr_pop["children_number"])
            assert children_number > 0
            rec_freq = float(curr_pop["rec_freq"])
            assert 0 <= rec_freq <= 1
            rec_rate = float(curr_pop["rec_rate"])
            assert 0 <= rec_rate <= 1
            partner_sel_strength = float(curr_pop["partner_sel_strength"])
            assert 0 <= partner_sel_strength <= 1
            has_sex = ast.literal_eval(curr_pop["has_sex"])
            if has_sex:
                male_number = int(curr_pop["male_number"])
                assert 0 < male_number < pop_size
                pop = SexualPopulation.generate_random( pop_size, mut_prob,
                    ti_prob, sel_strength, children_number, rec_freq,
                    rec_rate, partner_sel_strength, male_number, genome_len,
                    ppbw )
            else:
                pop = AsexualPopulation.generate_random( pop_size, mut_prob,
                    ti_prob, sel_strength, children_number, rec_freq,
                    rec_rate, partner_sel_strength, genome_len, ppbw )
            name = curr_pop["name"]
            populations[name] = pop
        environment = cls(populations, uoai, pbwsai, ppbw, apbw)
        return environment

    def evolve(self, iterations):
        switch_to_alternative = True
        for i in range(iterations):
            sys.stdout.write("\rIter: %s" % (i+1))
            sys.stdout.flush()
            if i+1 in self.pos_base_weights_switch_at_iter:
                if switch_to_alternative:
                    switch_to_alternative = False
                    new_pbw = self.alternative_pos_base_weights
                else:
                    switch_to_alternative = True
                    new_pbw = self.primary_pos_base_weights
                for pop in self.populations.values():
                    pop.update_pbw(new_pbw)
            if i+1 in self.update_origin_at_iter:
                for pop in self.populations.values():
                    pop.set_current_genome_as_origin()
            for pop_name, pop in self.populations.items():
                MONITOR.pop_name = pop_name
                pop.moran_step()
        print("")



def process_pbw_cfg_entry(pbw_cfg):
    pbw_dict = {}
    if pbw_cfg == "":
        return None
    all_p_indexes = []
    for l in pbw_cfg.split('\n'):
        no_space_l = ''.join(l.split())
        positions, base_weights = no_space_l.split(':')
        positions = positions.split(',')
        p_indexes = []
        for p in positions:
            if '-' in p:
                p_start, p_end = tuple(map(int, p.split('-')))
                p_indexes += list(range(p_start-1, p_end))
            else:
                p_indexes.append(int(p)-1)
        base_weights = {bw[0]:float(bw[1:]) for bw in base_weights.split(',')}
        for p in p_indexes:
            pbw_dict[p] = base_weights
        all_p_indexes += p_indexes
    assert list(sorted(pbw_dict)) == list(range(max(pbw_dict)+1))
    if len(all_p_indexes) != len(pbw_dict):
        print("WARNING! Position indexes overlap.")
    pbw = [bw for i, bw in sorted(pbw_dict.items(), key=lambda a:a[0])]
    return pbw

def process_int_list_cfg_entry(int_list_cfg):
    return list(map(int, int_list_cfg.split()))


def run_evolution(config):
    common = config["common"]
    print("Starting %s" % common["run_id"])
    environment = Environment.create_from_config(config)
    interations = int(common["interations"])
    environment.evolve(interations)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Run with: $python evolution.py config.cfg"
    cfg_file = sys.argv[1]
    assert os.path.isfile(cfg_file), "Config file not found"

    config = configparser.ConfigParser()
    config.read_file(open(cfg_file))
    runs_base_dir = config["common"]["runs_base_dir"]
    run_id = config["common"]["run_id"]
    run_dir = os.path.join(runs_base_dir, run_id)
    if os.path.isdir(run_dir):
        print("%s dir already exists" % run_dir)
        print("If you proceed its content will be overwritten")
        while True:
            answer = input("Do you want to proceed? (Y/N) --> ")
            if answer == "N":
                print("Terminating")
                sys.exit(0)
            elif answer == "Y":
                print("Cleaning %s" % run_dir)
                shutil.rmtree(run_dir)
                break
            else:
                print("Wrong input. Please choose either 'Y' or 'N'")
    os.makedirs(run_dir)
    shutil.copyfile(cfg_file, os.path.join(run_dir, "start.cfg"))
    log_file_name = os.path.join(run_dir, "run_log.txt")
    MONITOR = monitor.Monitor(log_file_name)
    run_evolution(config)
    MONITOR.close()
    print("Done")
