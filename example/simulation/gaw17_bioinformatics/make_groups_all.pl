#!/usr/bin/perl -w
use strict;

my $index=0;
my %genes=();
open(IN,"snp_info")|| die "Can't open snp info file\n";
<IN>;
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  print "$index\t$gene\n";
  ++$index;
}
close(IN);
