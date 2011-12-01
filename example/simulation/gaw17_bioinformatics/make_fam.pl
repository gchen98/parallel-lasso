#!/usr/bin/perl -w
use strict;

if (@ARGV<1){
  print "Usage: <replicates>\n"; 
  exit 1;
}
my $replicates=shift;

my $beta = log(2);

open(IN,"causal_genes")||die "Can't open causal genes file\n";
my %causal_genes=();
my %causal_snps=();
my $lastgene="";
my $bit=0;
srand(1);
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  #if ($lastgene ne $gene && rand()>.5){ $bit=!$bit; }
  $bit = rand()>.51;
  my $direction = $bit?1:-1;
  $causal_genes{$gene} = 1;
  $causal_snps{$snp} = $direction * $beta;
  $lastgene=$gene;
}
close(IN);

#my %effects=();
#open(IN,"positives.q1")|| die "Can't open true positives file\n";
#while(my $line=<IN>){
#  chomp($line);
#  my ($gene,$snp,$maf,$beta)=split(/\t/,$line);
#  $effects{$snp}=4*$beta;
#}
#close(IN);
#open(IN,"positives.q2")|| die "Can't open true positives file\n";
#while(my $line=<IN>){
#  chomp($line);
#  my ($gene,$snp,$maf,$beta)=split(/\t/,$line);
#  $effects{$snp}=-4*$beta;
#}
#close(IN);

my @effects=();
open(IN,"snp_info")||die "Can't open snpinfo file\n";
<IN>;
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  my $effect = $causal_snps{$snp};
  if (!defined($effect) ){ $effect = 0; } ;
  push(@effects,$effect);
}
close(IN);
my @templates=();
open(IN,"gaw17.fam.template")|| die "Can't open fam template file\n";
while(my $line=<IN>){
  chomp($line);
  push(@templates,$line);
}
close(IN);
for(my $replicate=0;$replicate<$replicates;++$replicate){
  print STDERR "Generating replicate $replicate\n";
  open(OUT,">famdata/gaw17.fam.".$replicate)||die "Can't open fam out\n";
  open(IN,"gaw17.geno")|| die "Can't open genotypes file\n";
  while(my $line=<IN>){
    chomp($line);
    my $risk=0;
    for(my $j=0;$j<length($line);++$j){
      my $g=substr($line,$j,1)-1;
      if ($g>0 && $effects[$j]!=0){
        $risk+=$effects[$j]*$g;
      }
    }
    my $prob = (exp($risk))/(1+exp($risk));
    #print STDERR "risk prob: $prob\n";
    my $aff=$prob>rand()?2:1;
    print OUT "$templates[$replicate]\t$aff\n";
  }
  close(IN);
  close(OUT);
}
 
