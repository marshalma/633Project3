#!/usr/local/bin/ruby

require 'csv'
require 'pry'
require 'matrix'

def preprocess(filename, class_value_index)
  raw_data = CSV.read(filename)
  d = []
  x = []
  class_values = {}
  raw_data.each do |item|
    class_values[item[class_value_index]] = 1 unless class_values.has_key? item[class_value_index]
    d << item[class_value_index]
    item.delete_at(class_value_index);
    tmp = []
    item.each do |a|
      tmp << a.to_f
    end
    x << tmp
  end
  return [d,x,class_values.keys]
end

def find_min_max(data)
  min_max = []
  n = data[0].size
  (0..n-1).each do |i|
    min = data[0][i]
    max = data[0][i]
    data.each do |item|
      min = item[i] if min > item[i]
      max = item[i] if max < item[i]
    end
    min_max << [min,max]
  end
  return min_max
end

def normalize!(data, min_max)
  min_max.each_with_index do |item, index|
    data.each {|d| d[index] = 0} if item[0] == item[1]
    mid = (item[0] + item[1])/2
    data.each do |d|
      d[index] = 2 * (d[index] - mid) / (item[1] - item[0])
    end
  end
end

def KNN(x, d, input, k)
  output = []
  input.each do |inp|
    dists = []
    x.each_with_index do |item,i|
      dists << [i, dist(item, inp)]
    end
    dists.sort! {|a,b| a[1]<=>b[1]}
    stat = {}
    (0..k-1).each do |i|
      if stat.has_key?(d[dists[i][0]])
        stat[d[dists[i][0]]] += 1
      else
        stat[d[dists[i][0]]] = 1
      end
    end
    max = -Float::INFINITY
    stat.values.each do |value|
      max = value if max < value
    end
    output << stat.key(max)
  end
  return output
end

def KNN_DW(x, d, input, k=x[0].size)
  output = []
  input.each do |inp|
    dists = []
    x.each_with_index do |item,i|
      dists<< [i, dist(item, inp)]
    end
    stat = {}
    (0..k-1).each do |i|
      if stat.has_key?(d[dists[i][0]])
        stat[d[dists[i][0]]] += 1/dists[i][1]**2
      else
        stat[d[dists[i][0]]] = 1/dists[i][1]**2
      end
    end
    max = -Float::INFINITY
    stat.values.each do |value|
      max = value if max < value
    end
    output << stat.key(max)
  end
  return output
end

def NTgrowth()
end

def PCA(x, t, threshold = 0.9)
  c = mean(x)
  trans_to_center! x,c
  trans_to_center! t,c
  x_mat = Matrix.rows(x)
  x_cov = x_mat.transpose * x_mat
  eigen_vec, eigen_mat = x_cov.eigen
  eigen_vec_a = eigen_vec.transpose.to_a
  eigen_values_with_vec = []
  eigen_mat.each_with_index :diagonal do |v,i|
    eigen_values_with_vec << [v, eigen_vec_a[i]]
  end

  eigen_values_with_vec.sort! {|a,b| b[0] <=> a[0]}
  sum_variance = 0.0
  eigen_values_with_vec.each do |item|
    sum_variance += item[0]
  end
  pc = []
  sum_cur = 0
  eigen_values_with_vec.each_with_index do |item|
    sum_cur += item[0]
    pc << item[1]
    break if (sum_cur >= threshold * sum_variance)
  end
  return pc
end

def dist(x1,x2)
  n = x1.size
  result = 0
  (0..n-1).each do |i|
    result += (x1[i] - x2[i])**2
  end
  return result
end

def accuracy(x1,x2)
  n = x1.size
  match = 0
  (0..n-1).each {|i| match += 1 if x1[i] == x2[i]}
  return match.to_f/n.to_f
end

def mean(x)
  n = x.size
  d = x[0].size
  result = []
  (0..d-1).each do |i|
    sum = 0
    x.each do |item|
      sum += item[i]
    end
    result << sum.to_f / n.to_f
  end
  return result
end

def trans_to_center!(x,c)
  x.each do |item|
    c.each_with_index do |v, i|
      item[i] -= v
    end
  end
end

def linear_trans(x,trans)
  return (Matrix.rows(trans) * Matrix.rows(x).transpose).transpose.to_a
end


# main proc
d,x,class_values = preprocess('./datasets/ForestTypes/training.csv', 0)
target,input = preprocess('./datasets/ForestTypes/testing.csv', 0)


min_max = find_min_max(x)
normalize!(x, min_max)
normalize!(input, min_max)


target_est = KNN(x,d,input,10)
# target_est = KNN_DW(x,d,input)
puts "KNN accuracy: " + accuracy(target, target_est).round(2).to_s


pca = PCA(x, input, 0.90)
x_tran = linear_trans(x,pca)
input_tran = linear_trans(input,pca)
min_max_tran = find_min_max(x_tran)

normalize! x_tran, min_max_tran
normalize! input_tran, min_max_tran

target_est = KNN(x_tran,d,input_tran,5)
puts "KNN with PCA accuracy: " + accuracy(target_est, target).round(2).to_s


# # tests
# # integral test for transformation to origin
# x = [[-1,-1,-1],[0,0,0],[1,1,1],[0,1,0]]
# trans_to_center! x, mean(x)
# puts x.to_s
#
# # unit test for PCA
# puts PCA(x).to_s
