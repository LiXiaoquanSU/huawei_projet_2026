"""Preserve initial search result; export repeated-stability calibration choice."""
from robust_search import *
def main():
    s=read(RWORK/'selection_repeated.json');selection=dict(base=s['base'],delta=s['delta'],eligible=s['eligible'],recipe=dict(radius=s['radius'],objective=s['objective']),selection_method=s['choice'],rooms={})
    for room in ('a1','c1','g1','e1','actual'):
        z,y,rows=data(room,s['base']);ix=old.split(room,rows)[2];selection['rooms'][room]=old.metrics(z+z.new_tensor(s['delta']),y,z,ix)
    assert selection['rooms']['actual']['correct']==s['source']['correct']
    if not (RWORK/'selection_initial.json').exists():save(RWORK/'selection_initial.json',read(RWORK/'selection.json'))
    save(RWORK/'selection.json',selection)
    import build_bias
    build_bias.BWORK=RWORK;build_bias.main()
if __name__=='__main__':main()
