# frozen_string_literal: true

#
# Topic Visualization index generator
#
# Feeds /visualization.html (layout: visualization) and the home page overview. Reads every
# transcript in _data/transcripts/ and, at build time, writes:
#
#   1. <output>/index.json - which lines of which interview carry which tag,
#      and each tag's colour.
#      No transcript text, so the page can draw the whole collection from one
#      small file.
#   2. <output>/<tag-slug>.json - the passages (line, timestamp, speaker,
#      words) for one tag, grouped by interview. The page fetches a tag's file
#      only when a reader selects that tag.
#
# Tags come from the "tags" column of each transcript row ("family;education").
# Spellings that slugify the same ("LCOH Places", "lcoh places") are merged.
# Tags in square brackets are processing notes, not topics, and are skipped.
#
# Optional settings in _data/theme.yml (defaults shown):
#   topic-viz-tags: filters              (filters -> only tags in _data/filters.csv | all)
#   topic-viz-output: assets/data/topic-viz
#   topic-viz-transcripts: transcripts   (folder in _data holding the transcript CSVs)
#   topic-viz-item-path: items           (folder of interview pages; auto-detected from page_gen)
#

require 'json'
require 'fileutils'

module TopicViz
  # Order of the values in each passage array written to the tag files.
  FIELDS = %w[line timestamp speaker words].freeze

  # Same colours, in the same order, as _includes/transcript/style/filter-style.html,
  # so a tag listed in _data/filters.csv has one colour here and on the interview pages.
  # Tags not in filters.csv continue through the list, most common first; with more
  # than 20 tags the colours repeat.
  PALETTE = %w[
    #e6194b #3cb44b #ffe119 #4363d8 #f58231 #911eb4 #46f0f0 #f032e6 #bcf60c #fabebe
    #008080 #e6beff #9a6324 #fffac8 #800000 #aaffc3 #808000 #ffd8b1 #000075 #808080
  ].freeze

  # Written straight to _site, skipping Liquid (transcript text may contain "{{").
  class GeneratedFile < Jekyll::StaticFile
    def initialize(site, dir, name, content)
      super(site, site.source, dir, name)
      @generated_content = content
    end

    def write(dest)
      dest_path = destination(dest)
      FileUtils.mkdir_p(File.dirname(dest_path))
      File.write(dest_path, @generated_content)
      true
    end

    def modified?
      true
    end

    def mtime
      Time.now.to_i
    end

    def modified_time
      Time.now
    end
  end

  class Generator < Jekyll::Generator
    safe true
    priority :lowest

    DEFAULTS = {
      'topic-viz-tags' => 'filters',
      'topic-viz-output' => 'assets/data/topic-viz',
      'topic-viz-transcripts' => 'transcripts'
    }.freeze

    def generate(site)
      theme = site.data['theme'].is_a?(Hash) ? site.data['theme'] : {}
      cfg = DEFAULTS.merge(theme.select { |key, value| DEFAULTS.key?(key) && !value.nil? })

      transcripts = site.data[cfg['topic-viz-transcripts']]
      unless transcripts.is_a?(Hash) && !transcripts.empty?
        Jekyll.logger.warn 'Topic Viz:', "no transcripts in _data/#{cfg['topic-viz-transcripts']}, skipping."
        return
      end

      filters = load_filters(site)
      only_filters = cfg['topic-viz-tags'].to_s.strip.downcase == 'filters'
      items, people = load_metadata(site)

      tags = {}       # slug => { spellings, lines (count), interviews, passages }
      interviews = [] # one entry per transcript, in id order
      skipped = Hash.new { |hash, key| hash[key] = [] }
      missing = []

      transcripts.keys.map(&:to_s).sort.each do |objectid|
        rows = transcripts[objectid]
        next unless rows.is_a?(Array)

        item = items[objectid]
        unless item
          missing << objectid
          next
        end

        by_tag = Hash.new { |hash, key| hash[key] = [] }
        tagged = []
        rows.each_with_index do |row, line|
          next unless row.respond_to?(:[])

          slugs = []
          row['tags'].to_s.split(';').map(&:strip).reject(&:empty?).each do |name|
            if name.start_with?('[')
              skipped[name] << objectid unless skipped[name].include?(objectid)
              next
            end
            slug = Jekyll::Utils.slugify(name)
            next if slug.to_s.empty? || slugs.include?(slug)
            next if only_filters && !filters.key?(slug)

            slugs << slug
            tag = tags[slug] ||= { 'spellings' => Hash.new(0), 'count' => 0, 'passages' => {} }
            tag['spellings'][name] += 1
            tag['count'] += 1
            by_tag[slug] << line
            (tag['passages'][objectid] ||= []) << [
              line, row['timestamp'].to_s.strip, row['speaker'].to_s.strip.sub(/:\z/, ''), row['words'].to_s.strip
            ]
          end
          tagged << line unless slugs.empty?
        end

        interviews << {
          'id' => objectid, 'title' => item['title'], 'person' => item['person'],
          'n' => rows.size, 'tagged' => tagged, 'by_tag' => by_tag
        }
      end

      publish(site, cfg, tags, interviews, people, filters)

      skipped.each do |name, ids|
        Jekyll.logger.warn 'Topic Viz:', "skipped tag #{name} found in #{ids.size} transcript(s): #{ids.join(', ')}"
      end
      unless missing.empty?
        Jekyll.logger.warn 'Topic Viz:', "no metadata row for #{missing.join(', ')}; left out of the visualization."
      end
    end

    private

    # slug => { 'tag', 'description' } from _data/filters.csv
    def load_filters(site)
      rows = site.data['filters']
      return {} unless rows.is_a?(Array)

      rows.each_with_object({}) do |row, filters|
        name = row['tag'].to_s.strip
        next if name.empty?

        slug = Jekyll::Utils.slugify(name)
        next if filters.key?(slug)

        filters[slug] = { 'tag' => name, 'description' => row['description'].to_s.strip, 'position' => filters.size }
      end
    end

    # items:  objectid => { 'title', 'person' (id of the row the interview is grouped under) }
    # people: id => display name
    def load_metadata(site)
      metadata = site.data[site.config['metadata'].to_s]
      items = {}
      people = {}
      return [items, people] unless metadata.is_a?(Array)

      metadata.each do |row|
        id = row['objectid'].to_s.strip
        next if id.empty?

        if row['display_template'].to_s.strip == 'person'
          name = row['person_name'].to_s.strip
          people[id] = name.empty? ? row['title'].to_s.strip : name
        end
      end

      metadata.each do |row|
        id = row['objectid'].to_s.strip
        next if id.empty? || row['display_template'].to_s.strip == 'person'

        title = row['title'].to_s.strip
        person = row['personid'].to_s.strip
        # An interview with no person record stands as its own row,
        # labelled by the interviewee when there is one.
        unless people.key?(person)
          person = id
          interviewee = row['interviewee'].to_s.strip
          people[id] ||= [interviewee, title, id].find { |name| !name.empty? }
        end
        items[id] = { 'title' => title.empty? ? id : title, 'person' => person }
      end
      [items, people]
    end

    def publish(site, cfg, tags, interviews, people, filters)
      out_dir = cfg['topic-viz-output'].to_s.gsub(%r{\A/+|/+\z}, '')

      ordered = tags.sort_by { |slug, tag| [-tag['count'], slug] }
      tag_index = {}
      next_color = filters.size
      tag_rows = ordered.each_with_index.map do |(slug, tag), index|
        tag_index[slug] = index
        filter = filters[slug]
        label = filter ? filter['tag'] : tag['spellings'].max_by { |name, count| [count, name] }.first
        tag['label'] = label
        position = filter ? filter['position'] : next_color
        next_color += 1 unless filter
        [slug, label, filter ? filter['description'] : '', tag['count'], tag['passages'].size,
         PALETTE[position % PALETTE.size]]
      end

      used_people = interviews.map { |interview| interview['person'] }.uniq
      person_index = used_people.each_with_index.to_h
      index = {
        'base' => item_base(site),
        'tags' => tag_rows,
        'people' => used_people.map { |id| [id, people[id].to_s.empty? ? id : people[id]] },
        'interviews' => interviews.map do |interview|
          [
            interview['id'], interview['title'], person_index[interview['person']], interview['n'],
            interview['tagged'],
            interview['by_tag'].to_h { |slug, lines| [tag_index[slug], lines] }
          ]
        end
      }
      site.static_files << GeneratedFile.new(site, out_dir, 'index.json', JSON.generate(index))

      ordered.each do |slug, tag|
        payload = { 'tag' => slug, 'label' => tag['label'], 'fields' => FIELDS, 'interviews' => tag['passages'] }
        site.static_files << GeneratedFile.new(site, out_dir, "#{slug}.json", JSON.generate(payload))
      end

      site.data['topic_viz'] = { 'tags' => tag_rows.size, 'interviews' => interviews.size, 'output' => out_dir }
      Jekyll.logger.info 'Topic Viz:', "#{tags.values.sum { |tag| tag['count'] }} tagged passages across " \
                                       "#{tag_rows.size} tags and #{interviews.size} interviews."
    end

    def item_base(site)
      dir = site.data.dig('theme', 'topic-viz-item-path').to_s.strip
      if dir.empty?
        generators = Array(site.config['page_gen']).select { |gen| gen.is_a?(Hash) }
        match = generators.find { |gen| gen['data'].to_s == site.config['metadata'].to_s } || generators.first
        dir = match ? match['dir'].to_s.strip : ''
      end
      dir = 'items' if dir.empty?
      "/#{dir.gsub(%r{\A/+|/+\z}, '')}/"
    end
  end
end
