#!/usr/bin/perl 
use strict;
use warnings;
use File::Basename;

my $bin = './bfs_ftr_prof';
my $dataset = '/mnt/188/b/bfs/com-orkut.dat';

for my $i (0 .. 7) {
    print `$bin\_$i $dataset`;
}