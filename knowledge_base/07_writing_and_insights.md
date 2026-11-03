---
title: Technical Writing, Insights and LinkedIn Posts
tags: articles, posts, blog, writing, insights, security, encryption, dpapi, vulnerability, api, graphql, leetcode, learn in public, linkedin post, opinion, views, sumit
aliases: post kya likha, article, blog, linkedin post, security wala post, vulnerability kya mili, leetcode solution, लेख, पोस्ट, सुरक्षा, dpapi kya hai, machine bound, data exposure, two sum, palindrome, parentheses
---

# Technical Writing, Insights and LinkedIn Posts

I write on LinkedIn about security engineering (machine-bound encryption, API data exposure), DSA problem solving (LeetCode Learn in Public), my projects, and my learning journey.

## Machine-Bound Encryption with Windows DPAPI — Protecting App Secrets

- Context: I built an internal service for my team (APIs, SDKs, auth tokens) and wanted to ship it as a standalone .exe for a small group of authorized users, without a full deployment and CI/CD pipeline.
- Problem: packaging as an .exe does not protect secrets. If the executable or config is shared with someone unauthorized, embedded credentials could be extracted.
- My solution: machine-bound encryption with Windows DPAPI (Data Protection API). CryptProtectData with the default user scope ties decryption to the same user's Windows credentials, so copying the file to another machine or account usually isn't enough to decrypt it.
- Takeaways: encryption at rest, machine/user binding, least privilege, and defense in depth.
- Caveat: it isn't foolproof. If the authorized account or the running app is compromised, secrets can still leak. Use limited-scope, short-lived, revocable credentials and possibly a backend credential broker.
- Why it matters: data under HIPAA and FERPA needs every layer of protection. Security means choosing the right mechanism for the threat, not the most complex algorithm.

## Excessive Data Exposure in a GraphQL API — My Responsible Disclosure Post

- While learning Low-Level Design on a well-known learning platform, I inspected network logs in DevTools (a debugging habit of mine) and found an excessive data exposure vulnerability in its backend API.
- The API returned far more than the UI used: SuperAdmin PII, internal user details, roles, and in some flows data belonging to other users.
- The GraphQL schema exposed 183 mutation names (write operations such as createRole, deleteArticle, updateTicket, toggleRolePermission).
- Lesson: security bugs often come from simple API design oversights, such as returning full database objects instead of stripped DTOs. Check exactly what each endpoint returns and enforce server-side authorization. Tags: #AppSec, #BrokenAccessControl, #ResponsibleDisclosure.

## LeetCode Learn in Public — Two Sum, Valid Palindrome, Valid Parentheses

I started a "Learn in Public" LeetCode series, thanking my mentor Suraj Mourya Sir:
- Day 1, Two Sum: brute force O(n²); two pointers after sorting O(n log n); optimal hash map complement lookup O(n) time, O(n) space. Example: nums = [2,7,11,15], target = 9 gives [0,1].
- Day 2, Valid Palindrome (LeetCode 125): instead of building a cleaned reversed string, I used two pointers moving inward, skipping non-alphanumeric characters and comparing lowercase. One pass, O(1) extra space. Example: "A man, a plan, a canal: Panama" is true.
- Day 3, Valid Parentheses (LeetCode 20): a stack. Push opening brackets; for closing brackets check the top matches, else return false; valid if the stack ends empty. Example: "()[]{}" is true.

## My Views on Ownership, Startups and AI Tools

- I reposted Anupam Mittal's post on taking ownership and an owner's mindset ("being an owner follows taking ownership").
- I praised the idea of white-labeling tech companies that help non-tech entrepreneurs start their own tech agency or product.
- In an early post I compared Google Bard with ChatGPT and preferred Bard for accuracy, detail, creativity and access to fresh web information (my opinion at that time).
- I've posted looking for a business partner to launch a software agency offering web development, app development, custom software and graphic design.
