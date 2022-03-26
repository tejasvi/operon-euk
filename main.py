from concurrent.futures import ThreadPoolExecutor
from functools import cache
from gzip import decompress
from heapq import heappop, heappush
from pprint import pprint
from random import shuffle
import re
from time import time

from utils import get_operon_data, req, req_json


def timed(f):
    def wrap(*a, **k):
        start = time()
        print(f"Running {f.__name__}")
        res = f(*a, **k)
        print(f"{f.__name__} took {time() - start}s")
        return res

    return wrap


@timed
@cache
def gene_proteins() -> dict[str, str]:
    fasta_pat = re.compile(
        r"(?:^|\n)>([^\n]*gene=([^\n ]*)[^\n]*)\n(.*?)(?=\n>|$)", flags=re.S
    )
    gene_protein_map = {}
    for match in fasta_pat.finditer(
        decompress(
            req(
                "https://downloads.wormbase.org/species/c_elegans/sequence/protein/c_elegans.canonical_bioproject.current.protein.fa.gz",
                None,
            )
        ).decode()
    ):
        *_, gene_id, p_seq = match.groups()
        if gene_id not in gene_protein_map:
            gene_protein_map[gene_id] = p_seq

        """
        Gene comprising the operon can have multiple isoforms therefore multiple possible protein sequnces.
        E.g. https://wormbase.org/species/c_elegans/operon/CEOP1412
        For C. elegans around 1/3 operonic genes have multiple isoforms
        Considering only "first" isoform.
        gene_protein_map.setdefault(gene_id, []).append(p_seq)

        95% proteins have size < 1024
        from collections import Counter
        for t in range(8, 12):
            hist = Counter(len(v) // 2**t for v in gene_protein_map.values())
            n = hist[0] / len(gene_protein_map)
            print(2**t, n, hist.items())
        """
    return gene_protein_map


@timed
@cache
def gene_pairs() -> list[tuple[str, str]]:
    gene_protein_map = gene_proteins()
    with ThreadPoolExecutor(max_workers=2) as ex:
        locations = {
            gene["feature_id"]: (
                gene["seqname"],
                gene["start"],
                gene["stop"],
            )
            for gene in ex.map(
                lambda gene: req_json(
                    f"https://wormbase.org/rest/widget/gene/{gene}/location?download=1&content-type=application%2Fjson",
                    None,
                )["fields"]["genomic_image"]["data"],
                gene_protein_map,
            )
        }

    pairs = []
    h = []
    chrom = None
    for cur_gene in sorted(gene_protein_map, key=lambda k: locations[k]):
        # Create pairs for adjacent and previously overlapping genes
        cur_chrom, cur_start, cur_stop = locations[cur_gene]
        adjacent_gene = None
        if chrom != cur_chrom:
            h = []
        while h:
            if h[0][0] < cur_start:
                adjacent_gene = heappop(h)
            else:
                break
        if adjacent_gene:
            heappush(h, adjacent_gene)
        pairs.extend((cur_gene, gene) for _, gene in h)
        heappush(h, (cur_stop, cur_gene))
        chrom = cur_chrom

    return pairs


@timed
@cache
def operons() -> dict[str, list[str]]:
    operon_ids = [
        o["wbid"]
        for o in req_json(
            "https://wormbase.org/search/operon/*?download=1&content-type=application%2Fjson",
            {"Referer": "https://wormbase.org/species/c_elegans/operon"},
        )["results"]
        if not (o["description"] and "deprecated" in o["description"].lower())
    ]

    gene_protein_map = gene_proteins()
    len(gene_protein_map)
    with ThreadPoolExecutor(max_workers=2) as ex:
        """
        Exclude pseudogenes which are not transcribed therefore protein sequence is absent.
        Skip operons with single remaining gene. Around 12 such in E. coli
        E.g. https://wormbase.org/species/c_elegans/gene/WBGene00014673
        """
        gene_operon_map = {
            gene: operon["fields"]["name"]["data"]["id"]
            for operon in ex.map(
                get_operon_data,
                operon_ids,
            )
            if len(
                genes := [
                    gene_name
                    for g in operon["fields"]["structure"]["data"]
                    if (gene_name := g["gene_info"]["id"]) in gene_protein_map
                ]
            )
            > 1
            for gene in genes
        }
    return gene_operon_map


@timed
@cache
def main():
    operon_map = operons()
    operonic = []
    non_operonic = []

    seen = {}

    for g1, g2 in gene_pairs():
        (
            operonic if operon_map.get(g1) == operon_map.get(g2, ...) else non_operonic
        ).append((g1, g2))
        if operon_map.get(g1) == operon_map.get(g2, ...):
            seen.setdefault(operon_map.get(g1), set()).update((g1, g2))

    rmap = {}
    for g, o in operon_map.items():
        rmap.setdefault(o, set()).add(g)
    # pprint(
    #     {
    #         o: [sorted(gs), sorted(seen.get(o, []))]
    #         for o, gs in rmap.items()
    #         if gs != seen.get(o)
    #     }
    # )

    shuffle(operonic)
    shuffle(non_operonic)

    pmap = gene_proteins()
    pf = lambda g: " ".join(pmap[g])
    data_x = [(pf(g1), pf(g2)) for g1, g2 in operonic + non_operonic[: len(operonic)]]
    data_y = [1] * (len(data_x) // 2) + [-1] * (len(data_x) // 2)

    temp = list(zip(data_x, data_y))
    shuffle(temp)
    data_x, data_y = zip(*temp)

    return data_x, data_y


if __name__ == "__main__":
    main()
