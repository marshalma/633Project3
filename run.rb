require 'csv'
require 'pry'

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
    data.each do |d|
      d[index] = (d[index] - item[0]) / (item[1] - item[0])
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
      stat[d[dists[i][0]]] = 1 if !(stat.has_key? d[dists[i][0]])
      stat[d[dists[i][0]]] += 1 if stat.has_key? d[dists[i][0]]
    end
    max = -Float::INFINITY
    stat.values.each do |value|
      max = value if max < value
    end
    output << stat.key(max)
  end
  return output
end

def KNN_DW(x, d, input, k)

end

def NTgrowth()
end

def PCA(x)
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

d,x,class_values = preprocess('./datasets/ForestTypes/training.csv', 0)
min_max = find_min_max(x)
normalize!(x, min_max)

target,input = preprocess('./datasets/ForestTypes/testing.csv', 0)
normalize!(input, min_max)
target_est = KNN(x,d,input,10)
puts target_est.to_s
puts target.to_s
puts accuracy(target, target_est)
