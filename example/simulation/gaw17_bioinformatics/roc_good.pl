#!/usr/bin/perl -w
use strict;
use Cwd;

if (@ARGV<4){
 print "Usage: <nullsnps><rsq><OR><scenario>\n";
 exit(1);
}
my $nullsnps=shift;
my $causal=shift;
my $or=shift;
my $scenario=shift;

my $replicates=100;

my $step=1;
#my $iterations=100;
my $iterations=450001;

my $tpfile="../true_positives.txt";
my $tnfile="../true_negatives.txt";
my %tphash=();
my %tnhash=();
open(IN,$tpfile)||die "Can't open $tpfile\n";
while(my $line=<IN>){
  chomp($line);
  my ($maineff,$z,$a)=split(/\t/,$line);
  $tphash{$maineff}=1;
}
close(IN);
open(IN,$tnfile)||die "Can't open $tnfile\n";
while(my $line=<IN>){
  chomp($line);
  my ($maineff,$z,$a)=split(/\t/,$line);
  $tnhash{$maineff}=1;
}
close(IN);
my $exptp = $replicates*scalar(keys(%tphash));
my $exptn = $replicates*scalar(keys(%tnhash));
#my $exptn = (10016*10015/2+10016)-$exptp;
#print STDERR "Expected positives $exptp  negatives $exptn\n";
my %sensitivityhash=();
my %specificityhash=();
my $basepath=getcwd;
use DBI;
my $dbh=DBI->connect('DBI:mysql:database=pathway_mcmc_aabc;host=mec:3306','garyc','m3teknik');

my $select=$dbh->prepare("create temporary table summary(maineff varchar(20),n int,freq float,beta float,se float,hm float,car float,prior float,alpha float,key(maineff)) type=heap");
$select->execute();
$select->finish;
$select=$dbh->prepare("load data infile '".$basepath."/nullsnps_".$nullsnps."_causal_".$causal."_or_".$or."_scenario.".$scenario."' into table summary");
$select->execute();
$select->finish;
my $select_sens=$dbh->prepare("select maineff,count(*),avg(alpha) from summary where n>? group by maineff order by count(*) desc");
#for(my $i=0;$i<100;++$i){
for(my $i=0;$i<$iterations;$i+=$step){
  $select_sens->execute($i);
  while(my @selectdata=$select_sens->fetchrow_array){
    my ($maineff,$count,$alpha)=@selectdata;
    my $tpcount=0;
    my $tncount=0;
    if (defined($tphash{$maineff})){
      my $oldcount=$sensitivityhash{$i};
      if (!defined($oldcount)){ $sensitivityhash{$i}=0; }
      $tpcount=$count;
      #print "$i $maineff $count\n";
      #print "pos i $i maineff $maineff: count $count alpha $alpha\n";
    }elsif (defined($tnhash{$maineff})){
      my $oldcount=$specificityhash{$i};
      if (!defined($oldcount)){ $specificityhash{$i}=0; }
      #print "neg i $i maineff $maineff: count $count alpha $alpha\n";
      $tncount=$count;
    }
    $sensitivityhash{$i}+=$tpcount;
    $specificityhash{$i}+=$tncount;
    #print STDERR "$i: ".join(" ",@selectdata)."\n";
  }
}
$select_sens->finish;
#print "ITERATIONS\tFALSE POSITIVE\tPOWER\n";
$dbh->disconnect;
for(my $i=0;$i<$iterations;$i+=$step){
  my $tp = $exptp?$sensitivityhash{$i}/$exptp:0;
  my $fp = $exptn?$specificityhash{$i}/$exptn:0;
  print "$i\t$fp\t$tp\n";
}

