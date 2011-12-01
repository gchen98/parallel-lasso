#!/usr/bin/perl -w
use strict;

my %chrs=();
my %positions=();
my %genes=();
my $beta=2;

open(IN,"causal_genes")||die "Can't open causal genes file\n";
my %causal_genes=();
my %causal_snps=();
my $lastgene="";
my $bit=0;
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  if ($lastgene ne $gene){ $bit=!$bit; }
  my $direction = $bit?1:-1;
  $causal_genes{$gene}=$direction*$beta;
  $causal_snps{$snp} = 1;
  $lastgene=$gene;
}
close(IN);


open(IN,"snp_info")|| die "Can't open snpinfo file\n";
<IN>;
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  my $causal_snp=$causal_snps{$snp};
  my $effect=$causal_genes{$gene};
  my $prefix=(defined($effect) && defined($causal_snp))?"CAUSAL":"NULL";
  print "$chr\t".$prefix."_".$gene."_".$snp."\t0\t$pos\tA\tC\n";
}

#my %effects=();
#open(IN,"positives")|| die "Can't open true positives file\n";
#while(my $line=<IN>){
#  chomp($line);
#  my ($gene,$snp,$maf,$beta)=split(/\t/,$line);
#  $effects{$snp}=$beta;
#}
#close(IN);
#my $index=0;
#while(my $snp=<STDIN>){
#  chomp($snp);
#  my $chr=$chrs{$snp};
#  my $gene=$genes{$snp};
#  my $pos=$positions{$snp};
#  my $effect=$effects{$snp};
#  my $prefix=(defined($effect))?"CAUSAL":"NULL";
#  print "$chr\t".$prefix."_".$gene."_".$snp."\t0\t$pos\tA\tC\n";
#  ++$index;
#}
