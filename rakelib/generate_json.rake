###############################################################################
# TASK: generate_json
#
# create json files from Oral History as Data transcripts
# requires ImageMagick installed!
###############################################################################

# filepath: Rakefile
require 'rake'
require 'yaml'
require 'json'
require 'digest'
require 'time' # For iso8601

# --- Configuration Constants ---
CONFIG_FILE = '_config.yml'
SOURCE_DIR = File.expand_path('..', File.dirname(__FILE__)) # Project root (one level up from rakelib)
DATA_DIR = File.join(SOURCE_DIR, '_data')
# Assuming transcript CSVs are in a subdirectory named 'transcripts' within _data
TRANSCRIPTS_DATA_DIR = File.join(DATA_DIR, 'transcripts')
# Target directory for individual JSONs within the source 'assets'
ASSETS_TARGET_DIR = File.join(SOURCE_DIR, 'assets', 'data', 'transcripts')
# Support these file types (from generate_derivatives, keep it global if potentially reusable)
EXTNAME_TYPE_MAP = {
  '.jpeg' => :image,
  '.jpg' => :image,
  '.pdf' => :pdf,
  '.png' => :image,
  '.tif' => :image,
  '.tiff' => :image
}.freeze


# --- Helper Function: Load CSV Data ---
# Reads a CSV file and returns an array of hashes, with symbol keys
def load_csv_data(filepath)
  data = []
  begin
    # Read with headers, convert headers to lowercase symbols
    CSV.foreach(filepath, headers: true, header_converters: :symbol, converters: :all) do |row|
      data << row.to_h
    end
    puts "Successfully loaded #{data.length} rows from #{filepath}"
  rescue Errno::ENOENT
    puts "Error: CSV file not found at #{filepath}"
  rescue => e
    puts "Error reading CSV file #{filepath}: #{e.message}"
  end
  data
end

# --- Rake Task: generate_json ---
desc "Generate individual JSON files from transcript CSVs"
task :generate_json do
  puts "Starting JSON generation task..."

  # 1. Load Jekyll Configuration
  config = {}
  if File.exist?(CONFIG_FILE)
    config = YAML.load_file(CONFIG_FILE) || {}
    puts "Loaded configuration from #{CONFIG_FILE}"
  else
    puts "Warning: #{CONFIG_FILE} not found. Using default settings."
  end

  # 2. Determine Metadata File Path
  metadata_filename = config['metadata'] # Get metadata filename key from _config.yml
  metadata_path = nil
  if metadata_filename
    # Assume metadata file is directly in _data and is a CSV
    metadata_path = File.join(DATA_DIR, "#{metadata_filename}.csv")
  else
    puts "Warning: 'metadata' key not found in #{CONFIG_FILE}. Metadata will not be loaded."
  end

  # 3. Load Metadata
  metadata_collection = []
  if metadata_path && File.exist?(metadata_path)
    metadata_collection = load_csv_data(metadata_path)
  elsif metadata_path
    puts "Warning: Metadata file specified but not found at #{metadata_path}"
  end

  # 4. Load Transcript Data from CSVs
  transcripts = {}
  if Dir.exist?(TRANSCRIPTS_DATA_DIR)
    Dir.glob(File.join(TRANSCRIPTS_DATA_DIR, '*.csv')).each do |csv_file|
      # Use the CSV filename (without extension) as the transcript ID
      transcript_name = File.basename(csv_file, '.csv')
      transcript_content = load_csv_data(csv_file)
      if transcript_content.any?
         transcripts[transcript_name] = transcript_content
      else
         puts "Warning: No data loaded from #{csv_file}, skipping."
      end
    end
  else
    puts "Error: Transcripts data directory not found at #{TRANSCRIPTS_DATA_DIR}"
    puts "Please ensure your transcript CSV files are located in #{TRANSCRIPTS_DATA_DIR}"
    exit(1) # Exit task with an error code
  end

  if transcripts.empty?
    puts "No transcript CSV files were successfully loaded from #{TRANSCRIPTS_DATA_DIR}. Exiting."
    exit(1) # Exit task with an error code
  end

  # 5. Ensure Target Directories Exist
  FileUtils.mkdir_p(ASSETS_TARGET_DIR)
  FileUtils.mkdir_p(DATA_DIR) # For collection file

  # 6. Prepare Collection Data Structure
  collection_data = {
    'metadata': {
      'title': 'Complete Oral History Collection',
      'description': config['description'] || 'Oral history transcripts',
      'date_generated': Time.now.utc.iso8601,
      'transcript_count': transcripts.keys.length
    },
    'transcripts': {}
  }

  # 7. Process Each Transcript
  puts "Processing #{transcripts.keys.length} transcripts..."
  transcripts.each do |transcript_name, transcript_data|
    puts "  Processing: #{transcript_name}"

    # Find corresponding metadata (using symbol :objectid key from CSV helper)
    metadata = metadata_collection.find { |item| item[:objectid]&.to_s == transcript_name } || {}

    # Build JSON structure for the individual transcript
    json_data = {
      'title' => metadata[:title] || transcript_name,
      'interviewee' => metadata[:interviewee] || metadata[:title] || transcript_name,
      'interviewer' => metadata[:interviewer],
      'date' => metadata[:date],
      'subjects' => metadata[:subject]&.to_s&.split(';')&.map(&:strip)&.reject(&:empty?), # Ensure string before split
      'segments' => []
    }

    # Add segments (using symbol keys from CSV helper)
    transcript_data.each_with_index do |item, index|
        tags = item[:tags]&.to_s&.split(';')&.compact&.map(&:strip)&.reject { |t| t.nil? || t.strip.empty? } || []
        json_data['segments'] << {
            'id' => "#{transcript_name}_#{index}",
            'index' => index,
            'speaker' => item[:speaker],
            'words' => item[:words],
            'tags' => tags,
            'timestamp' => item[:timestamp]
        }
    end

    # Add transcript metadata (using symbol keys from CSV helper)
    json_data['metadata'] = {
        'totalSegments' => transcript_data.length,
        'description' => metadata[:description],
        'location' => metadata[:location],
        'source' => metadata[:source]
    }

    # Add this transcript's full data to the collection object
    collection_data[:transcripts][transcript_name] = json_data

    # Write individual JSON file to the source assets directory
    individual_path = File.join(ASSETS_TARGET_DIR, "#{transcript_name}.json")
    begin
      File.open(individual_path, 'w') do |file|
        file.write(JSON.pretty_generate(json_data))
      end
      # puts "    Successfully wrote individual file: #{individual_path}" # Optional: reduce verbosity
    rescue => e
      puts "    Error writing individual file #{individual_path}: #{e.message}"
    end
  end
  puts "Finished processing individual transcripts."
end

# Optional: Make this the default task when running `rake`
# task default: :generate_json